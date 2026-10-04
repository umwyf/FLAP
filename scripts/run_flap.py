#!/usr/bin/env python
"""Run FLAP inference (planner -> failure-aware execution) on QA benchmarks.

Example::

    python scripts/run_flap.py --config configs/inference.yaml \\
        --datasets nq hotpotqa --output_dir outputs/flap-7b

Ablations (Table 3) are exposed as flags, e.g. ``--no_check``,
``--no_local_replanning``, ``--no_paraphrases``, ``--closed_injection``,
and the search budget (Table 4) as ``--search_budget``.
"""

import argparse
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import load_dataset_split, write_jsonl, read_jsonl  # noqa: E402
from flap.eval.metrics import exact_match, f1_score, evidence_hit, aggregate  # noqa: E402
from flap.executor import FLAPExecutor, FLAPConfig  # noqa: E402
from flap.llm import build_llm  # noqa: E402
from flap.plan import SearchPlan  # noqa: E402
from flap.planner import Planner, StaticPlanner  # noqa: E402
from flap.retrieval import build_retriever  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--plans_file", default=None,
                        help="JSONL with precomputed plans ({question, plan}) to skip the planner")
    parser.add_argument("--search_budget", type=int, default=None)
    parser.add_argument("--no_check", action="store_true")
    parser.add_argument("--no_local_replanning", action="store_true")
    parser.add_argument("--no_paraphrases", action="store_true")
    parser.add_argument("--closed_injection", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    datasets = args.datasets or cfg["eval"]["datasets"]
    output_dir = args.output_dir or cfg["eval"]["output_dir"]
    batch_size = args.batch_size or cfg["eval"].get("batch_size", 64)
    os.makedirs(output_dir, exist_ok=True)

    exec_cfg = FLAPConfig(**cfg.get("execution", {}))
    if args.search_budget is not None:
        exec_cfg.search_budget = args.search_budget
    if args.no_check:
        exec_cfg.inject_check = False
    if args.no_local_replanning:
        exec_cfg.local_replanning = False
    if args.no_paraphrases:
        exec_cfg.use_paraphrases = False
    if args.closed_injection:
        exec_cfg.open_ended_injection = False

    retriever = build_retriever(**cfg["retriever"])
    agent_llm = build_llm(**cfg["agent"])

    if args.plans_file:
        plans = {r["question"]: SearchPlan.from_dict(r["plan"]) for r in read_jsonl(args.plans_file)}
        planner = StaticPlanner(plans)
    else:
        planner_cfg = dict(cfg["planner"])
        gen = planner_cfg.pop("generation", {})
        if planner_cfg.get("model") == cfg["agent"].get("model") and planner_cfg.get("backend") == cfg["agent"].get("backend") \
                and not planner_cfg.get("adapter"):
            planner_llm = agent_llm
        else:
            planner_llm = build_llm(**planner_cfg)
        planner = Planner(planner_llm, **gen)

    executor = FLAPExecutor(agent_llm, retriever, planner=planner, config=exec_cfg)
    summary = {}
    for name in datasets:
        examples = load_dataset_split(name, cfg["eval"]["data_root"], split=args.split, max_examples=args.max_examples)
        print(f"[{name}] {len(examples)} examples")
        t0 = time.time()
        questions = [e.question for e in examples]
        plans = planner.batch_plan(questions)
        results = executor.run(questions, plans, batch_size=batch_size)
        records = []
        for ex, res in zip(examples, results):
            first_docs = [f'{d["title"]} {d["text"]}' for d in (res.retrieved[0] if res.retrieved else [])]
            records.append({
                "id": ex.id,
                "dataset": ex.dataset,
                "question": ex.question,
                "golden_answers": ex.golden_answers,
                "answer": res.answer,
                "em": exact_match(res.answer, ex.golden_answers),
                "f1": f1_score(res.answer, ex.golden_answers),
                "first_hit": evidence_hit(first_docs, ex.golden_answers) if first_docs else 0.0,
                "num_search_calls": res.num_search_calls,
                "num_turns": res.num_turns,
                "finish_reason": res.finish_reason,
                "plan": res.plan,
                "trajectory": res.trajectory,
                "trace": res.trace,
            })
        write_jsonl(os.path.join(output_dir, f"{name}.jsonl"), records)
        summary[name] = aggregate(records)
        summary[name]["seconds"] = round(time.time() - t0, 1)
        print(f"[{name}] {json.dumps(summary[name])}")
    if summary:
        summary["average"] = {
            "em": sum(v["em"] for v in summary.values()) / len(summary),
            "avg_search_calls": sum(v["avg_search_calls"] for v in summary.values()) / len(summary),
        }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
