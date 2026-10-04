from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: List[str] = field(default_factory=list)
    include_stop: bool = True  # keep the matched stop string in the output
    seed: Optional[int] = None

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


class LLM(ABC):
    """Minimal completion-style interface shared by all backends."""

    @abstractmethod
    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> List[str]:
        """Continue each raw prompt and return the generated continuations."""

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Render chat messages into the raw prompt string expected by ``generate``."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
        return self.generate([self.apply_chat_template(messages)], config)[0]


def truncate_at_stop(text: str, stop: Sequence[str], include_stop: bool = True) -> str:
    """Cut ``text`` at the earliest stop string (optionally keeping it)."""
    best = None
    for s in stop:
        idx = text.find(s)
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, s)
    if best is None:
        return text
    idx, s = best
    return text[: idx + len(s)] if include_stop else text[:idx]


def build_llm(backend: str, model: str, **kwargs) -> LLM:
    """Factory: ``backend`` in {"hf", "vllm", "openai"}."""
    backend = backend.lower()
    if backend == "hf":
        from flap.llm.hf_backend import HFBackend

        return HFBackend(model, **kwargs)
    if backend == "vllm":
        from flap.llm.vllm_backend import VLLMBackend

        return VLLMBackend(model, **kwargs)
    if backend == "openai":
        from flap.llm.openai_backend import OpenAIBackend

        return OpenAIBackend(model, **kwargs)
    raise ValueError(f"Unknown LLM backend: {backend}")
