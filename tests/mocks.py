"""Scripted LLM and retriever used to exercise the executor without GPUs."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from flap import tags
from flap.llm.base import LLM, GenerationConfig, truncate_at_stop
from flap.retrieval.base import Document, InMemoryRetriever

WEEZER_DOCS = [
    Document("1", "Buddy Holly (song)", "Buddy Holly is a song by the American rock band Weezer, released in 1994."),
    Document("2", "Undone - The Sweater Song", "Undone - The Sweater Song is a song by Weezer released as a single in 1994."),
    Document("3", "Weezer (Blue Album)", "Weezer, also known as the Blue Album, is the debut studio album by Weezer."),
    Document("4", "Jaws (film)", "Jaws is a 1975 American thriller film directed by Steven Spielberg."),
    Document("5", "Steven Spielberg", "Steven Spielberg is an American filmmaker. He married Kate Capshaw in 1991."),
]


def weezer_retriever() -> InMemoryRetriever:
    return InMemoryRetriever(WEEZER_DOCS)


class ScriptedAgent(LLM):
    """A deterministic stand-in for the search agent.

    * In the *query* phase it closes the injected ``<think>`` prefix and
      searches for the guidance sentence; with a bare ``<think>`` (plan
      exhausted) it answers with ``answer_fn(context)``.
    * In the *check* phase it emits ``<pass>`` iff ``evidence_ok(last_info)``.
    """

    def __init__(self, evidence_ok, answer_fn, answer_in_check_phase=False):
        self.evidence_ok = evidence_ok
        self.answer_fn = answer_fn
        self.calls: List[Dict] = []

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "<|user|>\n" + messages[-1]["content"] + "\n<|assistant|>\n"

    def _last_information(self, context: str) -> str:
        infos = tags.INFO_RE.findall(context)
        return infos[-1] if infos else ""

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> List[str]:
        outs = []
        for ctx in prompts:
            self.calls.append({"stop": list(config.stop), "temperature": config.temperature})
            if tags.PASS in config.stop:  # check phase
                m = re.search(re.escape(tags.CHECK_OPEN) + r"([^\n]*)$", ctx)
                guidance = m.group(1) if m else ""
                ok = self.evidence_ok(self._last_information(ctx))
                out = f"{guidance.strip()} Looking at the evidence... {tags.CHECK_CLOSE} {tags.PASS if ok else tags.FAIL}"
            elif tags.ANSWER_CLOSE in config.stop:  # answer phase
                out = f" {self.answer_fn(ctx)} {tags.ANSWER_CLOSE}"
            else:  # query phase
                tail = ctx[ctx.rfind(tags.THINK_OPEN):]
                guidance = tail[len(tags.THINK_OPEN):].strip()
                if not guidance or guidance.lower().startswith("i found out"):
                    out = f" I found out the answer. {tags.THINK_CLOSE}\n{tags.ANSWER_OPEN}"
                else:
                    query = guidance.rstrip(".")
                    out = f" Let me search. {tags.THINK_CLOSE}\n{tags.SEARCH_OPEN} {query} {tags.SEARCH_CLOSE}"
            outs.append(truncate_at_stop(out, config.stop, config.include_stop))
        return outs
