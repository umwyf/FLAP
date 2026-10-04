#!/usr/bin/env python
"""Collect search trajectories from an RL-trained search agent (Figure 2, step 1).

The agent is a Search-R1-style model (e.g. one trained with PPO following
Jin et al., 2025).  For every NQ / HotpotQA training question we sample
``--num_samples`` trajectories; successful ones are selected later by
``scripts/build_planner_data.py``.

Example::

    python scripts/collect_trajectories.py \\
        --agent_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \\
        --backend vllm --retriever_url http://127.0.0.1:8000 \\
        --data_root data --output outputs/trajectories.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import load_train_questions, write_jsonl  # noqa: E402
from flap.eval.metrics import exact_match  # noqa: E402
from flap.executor import SearchAgentExecutor, SearchAgentConfig  # noqa: E402
from flap.llm import build_llm  # noqa: E402
from flap.retrieval import HTTPRetriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_model", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa"])
    parser.add_argument("--max_per_dataset", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=4, help="trajectories sampled per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_search_calls", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    questions = load_train_questions(args.data_root, args.datasets, args.max_per_dataset)
    print(f"Loaded {len(questions)} training questions from {args.datasets}")

    llm = build_llm(args.backend, args.agent_model)
    cfg = SearchAgentConfig(top_k=args.top_k, max_search_calls=args.max_search_calls,
                            query_temperature=args.temperature)
    agent = SearchAgentExecutor(llm, HTTPRetriever(args.retriever_url), cfg)

    records = []
    for sample_idx in range(args.num_samples):
        results = agent.run([q.question for q in questions], batch_size=args.batch_size)
        for ex, res in zip(questions, results):
            records.append({
                "id": ex.id,
                "dataset": ex.dataset,
                "question": ex.question,
                "golden_answers": ex.golden_answers,
                "sample": sample_idx,
                "trajectory": res.trajectory,
                "answer": res.answer,
                "em": exact_match(res.answer, ex.golden_answers),
                "num_search_calls": res.num_search_calls,
                "retrieved": res.retrieved,
            })
        n_succ = sum(r["em"] for r in records)
        print(f"sample {sample_idx}: {n_succ:.0f} successful trajectories so far")
        write_jsonl(args.output, records)
    print(f"Wrote {len(records)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
