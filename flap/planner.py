"""Planner model wrapper: question -> failure-aware search plan."""

from __future__ import annotations

from typing import List, Optional, Sequence

from flap.llm.base import LLM, GenerationConfig
from flap.plan import SearchPlan, parse_plan
from flap.prompts import planner_prompt


class Planner:
    """Generate a :class:`SearchPlan` for each question with a fine-tuned planner LLM.

    The planner is trained with supervised fine-tuning on (question, plan)
    pairs (see ``scripts/train_planner.py``) and decoded greedily by default.
    """

    def __init__(
        self,
        llm: LLM,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_steps: Optional[int] = None,
    ):
        self.llm = llm
        self.config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        self.max_steps = max_steps

    def build_prompt(self, question: str) -> str:
        return self.llm.apply_chat_template([{"role": "user", "content": planner_prompt(question)}])

    def plan(self, question: str) -> SearchPlan:
        return self.batch_plan([question])[0]

    def batch_plan(self, questions: Sequence[str]) -> List[SearchPlan]:
        prompts = [self.build_prompt(q) for q in questions]
        outputs = self.llm.generate(prompts, self.config)
        plans = []
        for out in outputs:
            plan = parse_plan(out)
            if self.max_steps is not None:
                plan.steps = plan.steps[: self.max_steps]
            plans.append(plan)
        return plans


class StaticPlanner:
    """A planner that returns pre-computed plans (e.g. loaded from a JSONL file)."""

    def __init__(self, plans_by_question: dict):
        self.plans = plans_by_question

    def plan(self, question: str) -> SearchPlan:
        p = self.plans.get(question)
        if p is None:
            return SearchPlan(steps=[])
        return p if isinstance(p, SearchPlan) else SearchPlan.from_dict(p)

    def batch_plan(self, questions: Sequence[str]) -> List[SearchPlan]:
        return [self.plan(q) for q in questions]
