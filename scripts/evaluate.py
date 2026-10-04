#!/usr/bin/env python
"""Re-score output files produced by run_flap.py / run_search_agent.py.

Reports exact match, F1, average search calls and (if available) the
first-retrieval hit rate.  Example::

    python scripts/evaluate.py outputs/flap-7b/*.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import read_jsonl  # noqa: E402
from flap.eval.metrics import exact_match, f1_score, aggregate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    rows = []
    for path in args.files:
        records = []
        for r in read_jsonl(path):
            r["em"] = exact_match(r.get("answer"), r["golden_answers"])
            r["f1"] = f1_score(r.get("answer"), r["golden_answers"])
            records.append(r)
        s = aggregate(records)
        s["file"] = os.path.basename(path)
        rows.append(s)
    header = f"{'file':<28}{'n':>7}{'EM':>8}{'F1':>8}{'calls':>8}{'1st-hit':>9}"
    print(header)
    print("-" * len(header))
    for s in rows:
        hit = f"{s['first_retrieval_hit']:.1f}" if "first_retrieval_hit" in s else "-"
        print(f"{s['file']:<28}{s['n']:>7}{s['em']:>8.1f}{s['f1']:>8.1f}{s['avg_search_calls']:>8.2f}{hit:>9}")
    if len(rows) > 1:
        print("-" * len(header))
        print(f"{'average':<28}{'':>7}{sum(r['em'] for r in rows)/len(rows):>8.1f}"
              f"{sum(r['f1'] for r in rows)/len(rows):>8.1f}{sum(r['avg_search_calls'] for r in rows)/len(rows):>8.2f}")
    print(json.dumps(rows, indent=2) if os.environ.get("FLAP_EVAL_JSON") else "")


if __name__ == "__main__":
    main()
