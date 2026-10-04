"""Structured failure-aware search plans.

A plan ``P = {(g_1, c_1), ..., (g_K, c_K)}`` consists of retrieval objectives
``g_i`` (injected as ``<think>`` guidance) and check objectives ``c_i``
(injected as ``<check>`` guidance).  Each retrieval objective is paired with a
set of paraphrased alternatives ``A(g_i)`` used for failure-triggered local
replanning.  The textual serialization follows the planner SFT template
(Table 17 of the paper)::

    Step 1:
    <think> {retrieval objective for step 1}
    <check> {verification objective for step 1}
    Paraphrases:
    - {alternative retrieval objective 1}
    - {alternative retrieval objective 2}
    - {alternative retrieval objective 3}
    Step 2:
    ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from flap import tags

_STEP_SPLIT_RE = re.compile(r"^\s*(?:\*\*)?Step\s*(\d+)\s*[:.]?(?:\*\*)?\s*$", re.MULTILINE | re.IGNORECASE)
_THINK_RE = re.compile(
    re.escape(tags.THINK_OPEN) + r"(.*?)(?=" + re.escape(tags.CHECK_OPEN) + r"|Paraphrases\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_CHECK_RE = re.compile(
    re.escape(tags.CHECK_OPEN) + r"(.*?)(?=Paraphrases\s*:|" + re.escape(tags.THINK_OPEN) + r"|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_PARA_BLOCK_RE = re.compile(r"Paraphrases\s*:(.*)\Z", re.DOTALL | re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-–•*]|\d+[.)])\s*(.+?)\s*$")


def _clean(text: str) -> str:
    for closing in (tags.THINK_CLOSE, tags.CHECK_CLOSE):
        text = text.replace(closing, "")
    return " ".join(text.split())


@dataclass
class PlanStep:
    """One plan step: a retrieval objective, a check objective and alternatives."""

    think: str
    check: Optional[str] = None
    paraphrases: List[str] = field(default_factory=list)

    @property
    def has_check(self) -> bool:
        return bool(self.check)

    def alternative(self, idx: int) -> str:
        """Return the ``idx``-th alternative objective (cycling), or ``think`` if none."""
        if not self.paraphrases:
            return self.think
        return self.paraphrases[idx % len(self.paraphrases)]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PlanStep":
        return cls(think=d["think"], check=d.get("check"), paraphrases=list(d.get("paraphrases", [])))


@dataclass
class SearchPlan:
    steps: List[PlanStep] = field(default_factory=list)
    raw: Optional[str] = None  # raw planner output, if any

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, idx: int) -> PlanStep:
        return self.steps[idx]

    @property
    def num_retrieval_steps(self) -> int:
        return sum(1 for s in self.steps if s.has_check)

    def to_dict(self) -> Dict:
        return {"steps": [s.to_dict() for s in self.steps], "raw": self.raw}

    @classmethod
    def from_dict(cls, d: Dict) -> "SearchPlan":
        return cls(steps=[PlanStep.from_dict(s) for s in d.get("steps", [])], raw=d.get("raw"))

    def format(self) -> str:
        return format_plan(self)


def format_plan(plan: SearchPlan, num_paraphrases: Optional[int] = None) -> str:
    """Serialize a plan with the planner SFT template."""
    blocks = []
    for i, step in enumerate(plan.steps, start=1):
        lines = [f"Step {i}:", f"{tags.THINK_OPEN} {step.think}"]
        if step.check:
            lines.append(f"{tags.CHECK_OPEN} {step.check}")
        paras = step.paraphrases if num_paraphrases is None else step.paraphrases[:num_paraphrases]
        if paras:
            lines.append("Paraphrases:")
            lines.extend(f"- {p}" for p in paras)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def parse_plan(text: str) -> SearchPlan:
    """Parse planner output text into a :class:`SearchPlan`.

    The parser is lenient: closing tags are optional, step headers may be
    bold-formatted, bullets may use ``-``, ``–``, ``•`` or numbers, and a step
    without a ``<check>`` line is kept as a reasoning-only step (e.g. a final
    "I found out that ..." step).  Steps without a ``<think>`` line are dropped.
    """
    raw = text
    text = text.strip()
    if not text:
        return SearchPlan(steps=[], raw=raw)

    # Split into step blocks.  If there are no explicit headers, treat the
    # whole text as a single step.
    positions = [m for m in _STEP_SPLIT_RE.finditer(text)]
    if positions:
        blocks = []
        for i, m in enumerate(positions):
            start = m.end()
            end = positions[i + 1].start() if i + 1 < len(positions) else len(text)
            blocks.append(text[start:end])
    else:
        blocks = [text]

    steps: List[PlanStep] = []
    for block in blocks:
        think_m = _THINK_RE.search(block)
        if not think_m:
            continue
        think = _clean(think_m.group(1))
        if not think:
            continue
        check_m = _CHECK_RE.search(block)
        check = _clean(check_m.group(1)) if check_m else None
        paraphrases: List[str] = []
        para_m = _PARA_BLOCK_RE.search(block)
        if para_m:
            for line in para_m.group(1).splitlines():
                b = _BULLET_RE.match(line)
                if b:
                    p = _clean(b.group(1))
                    if p:
                        paraphrases.append(p)
        steps.append(PlanStep(think=think, check=check or None, paraphrases=paraphrases))
    return SearchPlan(steps=steps, raw=raw)


DEFAULT_CHECK_OBJECTIVE = "Have I obtained evidence that satisfies the current retrieval objective?"


def rule_based_plan(search_queries: List[str]) -> SearchPlan:
    """Rule-based trajectory-to-plan conversion (Table 20, *FLAP w/o frontier LLM*).

    Every search query of a successful trajectory becomes one plan step with
    a templated retrieval objective, a generic check objective, and the
    original query as the only paraphrase.
    """
    steps = [
        PlanStep(
            think=f"I need to search for: {q}",
            check=DEFAULT_CHECK_OBJECTIVE,
            paraphrases=[q],
        )
        for q in search_queries
    ]
    return SearchPlan(steps=steps)
