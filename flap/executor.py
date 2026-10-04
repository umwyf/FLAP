"""Failure-aware plan execution (Algorithm 1 of the paper).

Given a question ``q``, a plan ``P = {p_1, ..., p_K}`` produced by the planner
and an off-the-shelf search agent ``M_a``, the executor:

1. injects the ``<think>`` guidance of the current plan step as an *open-ended
   prefix* into the agent context and lets the agent generate a search query
   (or a final answer);
2. retrieves evidence ``x = E(a)`` and appends it between ``<information>`` tags;
3. injects the ``<check>`` guidance and lets the agent produce a verification
   result ``<pass>``/``<fail>``;
4. on ``<fail>`` and remaining per-step search budget ``B``, performs
   *failure-triggered local replanning*: the retrieval objective is replaced by a
   paraphrased alternative and the same step is revisited;
5. after the plan is exhausted, lets the agent produce the final answer.

The implementation is a batched state machine so that many questions can be
executed in lock-step with batched LLM generation and batched retrieval.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from flap import tags
from flap.llm.base import LLM, GenerationConfig
from flap.plan import PlanStep, SearchPlan
from flap.prompts import search_agent_prompt, think_prefix, check_prefix
from flap.retrieval.base import Document, Retriever


class Phase(str, enum.Enum):
    QUERY = "query"        # agent reasons and emits <search> or <answer>
    RETRIEVE = "retrieve"  # waiting for (batched) retrieval of a pending query
    CHECK = "check"        # agent verifies evidence and emits <pass>/<fail>
    ANSWER = "answer"      # agent completes "<answer> ... </answer>" greedily
    DONE = "done"


@dataclass
class FLAPConfig:
    # --- Algorithm 1 ---
    search_budget: int = 2             # B: max search attempts per plan step (1 disables replanning)
    top_k: int = 3                     # passages per search call
    max_passage_words: Optional[int] = None  # truncate passages (None = retriever/server handles it)
    # --- ablation switches (Table 3) ---
    inject_check: bool = True          # w/o <check>
    local_replanning: bool = True      # w/o local replanning
    use_paraphrases: bool = True       # w/o paraphrased objectives (reuse original objective instead)
    open_ended_injection: bool = True  # w/o open-ended injection (inject closed segments instead)
    # --- decoding (Table 11) ---
    query_temperature: float = 0.7
    query_top_p: float = 0.95
    answer_temperature: float = 0.0
    check_temperature: float = 0.0
    max_new_tokens: int = 512          # per generated segment
    max_context_tokens: int = 4096
    # --- safety limits ---
    max_extra_turns: int = 2           # unguided search rounds allowed after the plan is exhausted
    max_total_turns: int = 12          # hard cap on LLM calls per question
    default_verification: str = "pass" # used when the agent emits neither <pass> nor <fail>


@dataclass
class ExecutionState:
    question: str
    plan: SearchPlan
    context: str                       # full prompt: chat prefix + assistant-side trajectory
    prefix_len: int                    # length of the chat prefix (to recover the trajectory)
    phase: Phase = Phase.QUERY
    step_idx: int = 0
    b: int = 0                         # search attempts used in the current step
    alt_idx: int = 0                   # next paraphrase index for the current step
    extra_turns: int = 0
    num_turns: int = 0
    num_search_calls: int = 0
    pending_query: Optional[str] = None
    answer: Optional[str] = None
    finish_reason: Optional[str] = None
    retrieved: List[List[Document]] = field(default_factory=list)
    trace: List[Dict] = field(default_factory=list)

    @property
    def current_step(self) -> Optional[PlanStep]:
        if 0 <= self.step_idx < len(self.plan.steps):
            return self.plan.steps[self.step_idx]
        return None

    @property
    def trajectory(self) -> str:
        return self.context[self.prefix_len:]


@dataclass
class ExecutionResult:
    question: str
    answer: Optional[str]
    trajectory: str
    plan: Dict
    num_search_calls: int
    num_turns: int
    finish_reason: Optional[str]
    retrieved: List[List[Dict]]
    trace: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)


class _AgentLoop:
    """Shared machinery for the plain search agent and the FLAP executor."""

    def __init__(self, agent_llm: LLM, retriever: Retriever, config: FLAPConfig):
        self.llm = agent_llm
        self.retriever = retriever
        self.cfg = config
        self.query_cfg = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.query_temperature,
            top_p=config.query_top_p,
            stop=[tags.SEARCH_CLOSE, tags.ANSWER_OPEN, tags.INFO_OPEN],
        )
        self.answer_cfg = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.answer_temperature,
            top_p=1.0,
            stop=[tags.ANSWER_CLOSE],
        )
        self.check_cfg = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.check_temperature,
            top_p=1.0,
            stop=[tags.PASS, tags.FAIL, tags.SEARCH_OPEN, tags.ANSWER_OPEN, tags.THINK_OPEN, tags.INFO_OPEN],
        )

    # ------------------------------------------------------------------ utils
    def _chat_prefix(self, question: str) -> str:
        return self.llm.apply_chat_template([{"role": "user", "content": search_agent_prompt(question)}])

    def _new_state(self, question: str, plan: SearchPlan) -> ExecutionState:
        prefix = self._chat_prefix(question)
        return ExecutionState(question=question, plan=plan, context=prefix, prefix_len=len(prefix))

    def _finish(self, s: ExecutionState, answer: Optional[str], reason: str):
        s.answer = answer
        s.finish_reason = reason
        s.phase = Phase.DONE
        s.pending_query = None

    def _force_answer(self, s: ExecutionState, reason: str):
        s.trace.append({"event": "force_answer", "reason": reason})
        if not s.context.rstrip().endswith(tags.ANSWER_OPEN):
            s.context += f"\n{tags.ANSWER_OPEN}"
        s.phase = Phase.ANSWER

    def _context_exhausted(self, s: ExecutionState) -> bool:
        return self.llm.count_tokens(s.context) + self.cfg.max_new_tokens > self.cfg.max_context_tokens

    def _to_result(self, s: ExecutionState) -> ExecutionResult:
        return ExecutionResult(
            question=s.question,
            answer=s.answer,
            trajectory=s.trajectory,
            plan=s.plan.to_dict(),
            num_search_calls=s.num_search_calls,
            num_turns=s.num_turns,
            finish_reason=s.finish_reason,
            retrieved=[[d.to_dict() for d in docs] for docs in s.retrieved],
            trace=s.trace,
        )

    # ------------------------------------------------------------ transitions
    def _on_query_output(self, s: ExecutionState, out: str):
        s.num_turns += 1
        stripped = out.rstrip()
        if stripped.endswith(tags.INFO_OPEN):  # agent tried to hallucinate evidence
            out = stripped[: -len(tags.INFO_OPEN)]
            s.context += out
            s.trace.append({"event": "malformed", "detail": "agent emitted <information>"})
            self._force_answer(s, "malformed_information")
            return
        s.context += out
        if stripped.endswith(tags.ANSWER_OPEN):
            s.phase = Phase.ANSWER
            return
        ans = tags.extract_answer(out)
        if ans is not None:
            self._finish(s, ans, "answer")
            return
        query = tags.extract_search_query(out)
        if query:
            if s.num_turns >= self.cfg.max_total_turns:
                self._force_answer(s, "max_total_turns")
                return
            s.pending_query = query
            s.phase = Phase.RETRIEVE
            return
        # Neither a search nor an answer (e.g. max_new_tokens hit).
        s.trace.append({"event": "malformed", "detail": "no <search> or <answer> in output"})
        self._force_answer(s, "malformed_query")

    def _on_answer_output(self, s: ExecutionState, out: str):
        s.num_turns += 1
        s.context += out
        ans = tags.extract_answer(s.context)
        if ans is None:  # unclosed answer: take what was generated
            ans = out.replace(tags.ANSWER_CLOSE, "").strip()
        self._finish(s, ans, s.finish_reason or "answer")

    def _append_information(self, s: ExecutionState, docs: List[Document]):
        s.num_search_calls += 1
        s.b += 1
        s.retrieved.append(docs)
        body = tags.format_information(docs, self.cfg.max_passage_words)
        s.context += f"\n\n{tags.INFO_OPEN}{body}{tags.INFO_CLOSE}\n\n"
        s.pending_query = None

    def _retrieve_pending(self, states: Sequence[ExecutionState]):
        pending = [s for s in states if s.phase == Phase.RETRIEVE and s.pending_query is not None]
        if not pending:
            return
        results = self.retriever.batch_search([s.pending_query for s in pending], top_k=self.cfg.top_k)
        for s, docs in zip(pending, results):
            s.trace.append({"event": "search", "step": s.step_idx, "query": s.pending_query,
                            "doc_ids": [d.id for d in docs]})
            self._append_information(s, docs)
            self._after_retrieval(s)

    def _after_retrieval(self, s: ExecutionState):  # overridden by FLAPExecutor
        self._inject_think(s, None)

    def _inject_think(self, s: ExecutionState, objective: Optional[str]):
        if objective is None:
            s.context += f"{tags.THINK_OPEN}"
        elif self.cfg.open_ended_injection:
            s.context += think_prefix(objective)
        else:
            s.context += f"{think_prefix(objective)} {tags.THINK_CLOSE}\n"
        s.phase = Phase.QUERY

    # ------------------------------------------------------------------ driver
    def _run_states(self, states: List[ExecutionState]) -> List[ExecutionResult]:
        while True:
            active = [s for s in states if s.phase != Phase.DONE]
            if not active:
                break
            for s in active:
                if s.phase in (Phase.QUERY, Phase.CHECK) and self._context_exhausted(s):
                    self._force_answer(s, "context_length")
            for phase, cfg, handler in (
                (Phase.QUERY, self.query_cfg, self._on_query_output),
                (Phase.CHECK, self.check_cfg, self._on_check_output),
                (Phase.ANSWER, self.answer_cfg, self._on_answer_output),
            ):
                group = [s for s in active if s.phase == phase]
                if not group:
                    continue
                outputs = self.llm.generate([s.context for s in group], cfg)
                for s, out in zip(group, outputs):
                    handler(s, out)
            self._retrieve_pending(active)
        return [self._to_result(s) for s in states]

    def _on_check_output(self, s: ExecutionState, out: str):  # pragma: no cover - plain agent has no checks
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Plain search agent (Search-R1-style inference, used for baselines and for
# trajectory collection when building planner data).
# ---------------------------------------------------------------------------
@dataclass
class SearchAgentConfig(FLAPConfig):
    max_search_calls: int = 4


class SearchAgentExecutor(_AgentLoop):
    """Standard ``<think>``–``<search>``–``<information>``–``<answer>`` agent loop without planning."""

    def __init__(self, agent_llm: LLM, retriever: Retriever, config: Optional[SearchAgentConfig] = None):
        super().__init__(agent_llm, retriever, config or SearchAgentConfig())

    def _after_retrieval(self, s: ExecutionState):
        if s.num_search_calls >= self.cfg.max_search_calls:
            self._force_answer(s, "max_search_calls")
        else:
            s.phase = Phase.QUERY  # the RL-trained agent emits <think> by itself

    def run(self, questions: Sequence[str], batch_size: int = 64, show_progress: bool = True) -> List[ExecutionResult]:
        results: List[ExecutionResult] = []
        iterator = range(0, len(questions), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="search-agent", unit="batch")
        for start in iterator:
            chunk = questions[start : start + batch_size]
            states = [self._new_state(q, SearchPlan(steps=[])) for q in chunk]
            results.extend(self._run_states(states))
        return results


# ---------------------------------------------------------------------------
# FLAP executor
# ---------------------------------------------------------------------------
class FLAPExecutor(_AgentLoop):
    def __init__(self, agent_llm: LLM, retriever: Retriever, planner=None, config: Optional[FLAPConfig] = None):
        super().__init__(agent_llm, retriever, config or FLAPConfig())
        self.planner = planner

    # -- plan step bookkeeping ------------------------------------------------
    def _start_step(self, s: ExecutionState):
        step = s.current_step
        s.b = 0
        s.alt_idx = 0
        if step is not None:
            s.trace.append({"event": "plan_step", "step": s.step_idx, "think": step.think, "check": step.check})
            self._inject_think(s, step.think)
            return
        # Plan exhausted: let the agent conclude (it may still search a bounded number of times).
        if s.extra_turns >= self.cfg.max_extra_turns or s.num_turns >= self.cfg.max_total_turns:
            self._force_answer(s, "plan_exhausted")
            return
        s.extra_turns += 1
        s.trace.append({"event": "unguided_turn", "n": s.extra_turns})
        self._inject_think(s, None)

    def _next_step(self, s: ExecutionState):
        s.step_idx += 1
        self._start_step(s)

    def _after_retrieval(self, s: ExecutionState):
        step = s.current_step
        if self.cfg.inject_check and step is not None and step.has_check:
            if self.cfg.open_ended_injection:
                s.context += check_prefix(step.check)
            else:
                s.context += f"{check_prefix(step.check)} {tags.CHECK_CLOSE}"
            s.phase = Phase.CHECK
        else:
            self._next_step(s)

    def _on_check_output(self, s: ExecutionState, out: str):
        s.num_turns += 1
        stripped = out.rstrip()
        for t in (tags.SEARCH_OPEN, tags.ANSWER_OPEN, tags.THINK_OPEN, tags.INFO_OPEN):
            if stripped.endswith(t):  # agent skipped the verdict; drop the dangling tag
                out = stripped[: -len(t)]
                break
        s.context += out
        verdict = tags.parse_verification(out)
        if verdict is None:
            verdict = self.cfg.default_verification
            s.context += f" {tags.PASS if verdict == 'pass' else tags.FAIL}"
        s.trace.append({"event": "check", "step": s.step_idx, "attempt": s.b, "verdict": verdict})
        step = s.current_step
        if (
            verdict == "fail"
            and self.cfg.local_replanning
            and step is not None
            and s.b < self.cfg.search_budget
            and s.num_turns < self.cfg.max_total_turns
        ):
            objective = step.alternative(s.alt_idx) if self.cfg.use_paraphrases else step.think
            s.alt_idx += 1
            s.trace.append({"event": "local_replan", "step": s.step_idx, "objective": objective})
            s.context += "\n"
            self._inject_think(s, objective)
            return
        s.context += "\n"
        self._next_step(s)

    # -- public API -------------------------------------------------------------
    def run(
        self,
        questions: Sequence[str],
        plans: Optional[Sequence[SearchPlan]] = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> List[ExecutionResult]:
        if plans is None:
            if self.planner is None:
                raise ValueError("Either pass `plans` or construct the executor with a planner.")
            plans = self.planner.batch_plan(questions)
        if len(plans) != len(questions):
            raise ValueError("`plans` must have the same length as `questions`.")
        results: List[ExecutionResult] = []
        iterator = range(0, len(questions), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="FLAP", unit="batch")
        for start in iterator:
            chunk_q = questions[start : start + batch_size]
            chunk_p = plans[start : start + batch_size]
            states = []
            for q, p in zip(chunk_q, chunk_p):
                s = self._new_state(q, p)
                self._start_step(s)
                states.append(s)
            results.extend(self._run_states(states))
        return results

    def run_one(self, question: str, plan: Optional[SearchPlan] = None) -> ExecutionResult:
        return self.run([question], [plan] if plan is not None else None, batch_size=1, show_progress=False)[0]
