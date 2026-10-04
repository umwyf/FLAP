#!/usr/bin/env python
"""Run a plain Search-R1-style search agent (no planning) as a baseline.

Example::

    python scripts/run_search_agent.py --config configs/inference.yaml \\
        --agent_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \\
        --datasets nq --output_dir outputs/search-r1-7b
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import load_dataset_split, write_jsonl  # noqa: E402
from flap.eval.metrics import exact_match, f1_score, aggregate  # noqa: E402
from flap.executor import SearchAgentExecutor, SearchAgentConfig  # noqa: E402
from flap.llm import build_llm  # noqa: E402
from flap.retrieval import build_retriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--agent_model", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_search_calls", type=int, default=4)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    agent_cfg = dict(cfg["agent"])
    if args.agent_model:
        agent_cfg["model"] = args.agent_model
    llm = build_llm(**agent_cfg)
    retriever = build_retriever(**cfg["retriever"])
    exec_cfg = SearchAgentConfig(max_search_calls=args.max_search_calls,
                                 top_k=cfg.get("execution", {}).get("top_k", 3))
    agent = SearchAgentExecutor(llm, retriever, exec_cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for name in args.datasets or cfg["eval"]["datasets"]:
        examples = load_dataset_split(name, cfg["eval"]["data_root"], max_examples=args.max_examples)
        results = agent.run([e.question for e in examples], batch_size=cfg["eval"].get("batch_size", 64))
        records = [{
            "id": ex.id, "question": ex.question, "golden_answers": ex.golden_answers,
            "answer": r.answer, "em": exact_match(r.answer, ex.golden_answers),
            "f1": f1_score(r.answer, ex.golden_answers), "num_search_calls": r.num_search_calls,
            "trajectory": r.trajectory,
        } for ex, r in zip(examples, results)]
        write_jsonl(os.path.join(args.output_dir, f"{name}.jsonl"), records)
        summary[name] = aggregate(records)
        print(f"[{name}] {json.dumps(summary[name])}")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
