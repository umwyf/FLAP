"""HuggingFace ``transformers`` backend (optionally with a PEFT/LoRA adapter)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from flap.llm.base import LLM, GenerationConfig, truncate_at_stop


class _StopOnStrings(StoppingCriteria):
    """Stop when every sequence in the batch contains one of the stop strings."""

    def __init__(self, tokenizer, stop: Sequence[str], prompt_lens: List[int]):
        self.tokenizer = tokenizer
        self.stop = list(stop)
        self.prompt_lens = prompt_lens
        self.done = [False] * len(prompt_lens)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stop:
            return False
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            new_ids = input_ids[i, self.prompt_lens[i]:]
            # Only decode the tail for efficiency.
            tail = self.tokenizer.decode(new_ids[-32:], skip_special_tokens=True)
            if any(s in tail for s in self.stop):
                self.done[i] = True
        return all(self.done)


class HFBackend(LLM):
    def __init__(
        self,
        model: str,
        adapter: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_batch_size: int = 16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=trust_remote_code
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.max_batch_size = max_batch_size

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> List[str]:
        outputs: List[str] = []
        for start in range(0, len(prompts), self.max_batch_size):
            batch = list(prompts[start : start + self.max_batch_size])
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]
            stopping = StoppingCriteriaList(
                [_StopOnStrings(self.tokenizer, config.stop, [prompt_len] * len(batch))]
            )
            gen_kwargs = dict(
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping,
            )
            if config.do_sample:
                gen_kwargs.update(temperature=config.temperature, top_p=config.top_p)
            if config.seed is not None:
                torch.manual_seed(config.seed)
            out = self.model.generate(**enc, **gen_kwargs)
            for row in out[:, prompt_len:]:
                text = self.tokenizer.decode(row, skip_special_tokens=True)
                outputs.append(truncate_at_stop(text, config.stop, config.include_stop))
        return outputs
