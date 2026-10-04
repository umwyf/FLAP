"""OpenAI-compatible chat backend (used for the frontier LLM in data construction).

Works with the OpenAI API as well as any OpenAI-compatible server (e.g.
``vllm serve``) via ``base_url``.  This backend cannot continue a raw prompt,
so it is *not* usable as the search-agent executor model.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

from flap.llm.base import LLM, GenerationConfig, truncate_at_stop


class OpenAIBackend(LLM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        system_prompt: Optional[str] = None,
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.max_retries = max_retries
        self.system_prompt = system_prompt

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        raise NotImplementedError("OpenAIBackend is chat-only; use chat() instead of raw prompts.")

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> List[str]:
        # Treat each raw prompt as a single user message.
        return [self.chat([{"role": "user", "content": p}], config) for p in prompts]

    def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
        if self.system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": self.system_prompt}] + list(messages)
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stop=list(config.stop) or None,
                    seed=config.seed,
                )
                text = resp.choices[0].message.content or ""
                return truncate_at_stop(text, config.stop, config.include_stop)
            except Exception as e:  # pragma: no cover - network errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        return ""
