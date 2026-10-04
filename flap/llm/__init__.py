"""Language-model backends.

All backends expose the same minimal completion-style interface
(:class:`flap.llm.base.LLM`).  The search-agent executor needs *raw-prompt
continuation* so that planner guidance can be injected as an open-ended prefix
inside the assistant turn; use :class:`HFBackend` or :class:`VLLMBackend` for
that.  :class:`OpenAIBackend` is a chat-only backend used for the frontier LLM
during planner data construction.
"""

from flap.llm.base import LLM, GenerationConfig, build_llm

__all__ = ["LLM", "GenerationConfig", "build_llm"]
