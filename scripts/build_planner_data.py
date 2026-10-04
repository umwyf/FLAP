#!/usr/bin/env python
"""Convert successful trajectories into planner SFT data (Figure 2, steps 2-3).

Example (frontier LLM, Table 19)::

    python scripts/build_planner_data.py --trajectories outputs/trajectories.jsonl \\
        --converter frontier --frontier_model gpt-4o-mini --output_dir data/planner

Example (rule-based, Table 20; *FLAP w/o frontier LLM*)::

    python scripts/build_planner_data.py --trajectories outputs/trajectories.jsonl \\
        --converter rule --output_dir data/planner_rule
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import read_jsonl, write_jsonl  # noqa: E402
from flap.data.planner_data import (  # noqa: E402
    FrontierPlanConverter, RuleBasedPlanConverter, build_planner_examples, plan_statistics, split_by_question,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, help="JSONL from scripts/collect_trajectories.py")
    parser.add_argument("--converter", choices=["frontier", "rule"], default="frontier")
    parser.add_argument("--frontier_model", default="gpt-4o-mini")
    parser.add_argument("--frontier_base_url", default=None, help="OpenAI-compatible endpoint (e.g. vllm serve)")
    parser.add_argument("--num_paraphrases", type=int, default=3)
    parser.add_argument("--min_paraphrases", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.trajectories))
    if args.max_records:
        records = records[: args.max_records]
    # Sort so that the first successful sample per question is kept (deterministic).
    records.sort(key=lambda r: (str(r.get("id", r["question"])), r.get("sample", 0)))
    print(f"Loaded {len(records)} raw trajectories")

    if args.converter == "frontier":
        from flap.llm.openai_backend import OpenAIBackend

        llm = OpenAIBackend(args.frontier_model, base_url=args.frontier_base_url)
        converter = FrontierPlanConverter(llm, num_paraphrases=args.num_paraphrases)
        min_paraphrases = args.min_paraphrases
    else:
        converter = RuleBasedPlanConverter()
        min_paraphrases = 0

    examples = build_planner_examples(records, converter, batch_size=args.batch_size,
                                      min_paraphrases=min_paraphrases)
    train, val = split_by_question(examples, val_ratio=args.val_ratio, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "train.jsonl"), (e.to_dict() for e in train))
    write_jsonl(os.path.join(args.output_dir, "val.jsonl"), (e.to_dict() for e in val))
    stats = {"train": plan_statistics(train), "val": plan_statistics(val)}
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
