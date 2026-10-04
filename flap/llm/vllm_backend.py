"""vLLM backend for fast batched inference."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from flap.llm.base import LLM, GenerationConfig, truncate_at_stop


class VLLMBackend(LLM):
    def __init__(
        self,
        model: str,
        adapter: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        **engine_kwargs,
    ):
        from vllm import LLM as _VLLM
        from transformers import AutoTokenizer

        self.adapter = adapter
        self.lora_request = None
        if adapter:
            from vllm.lora.request import LoRARequest

            engine_kwargs.setdefault("enable_lora", True)
            engine_kwargs.setdefault("max_lora_rank", 64)
            self.lora_request = LoRARequest("flap_adapter", 1, adapter)
        self.engine = _VLLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **engine_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> List[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p if config.do_sample else 1.0,
            stop=list(config.stop) or None,
            include_stop_str_in_output=config.include_stop,
            seed=config.seed,
        )
        results = self.engine.generate(list(prompts), params, use_tqdm=False, lora_request=self.lora_request)
        texts = [r.outputs[0].text for r in results]
        return [truncate_at_stop(t, config.stop, config.include_stop) for t in texts]
