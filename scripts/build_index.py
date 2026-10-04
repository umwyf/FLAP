#!/usr/bin/env python
"""Encode the wiki-18 corpus with E5 and build a FAISS inner-product index.

Example::

    python scripts/build_index.py --corpus_path data/wiki-18.jsonl --index_path indexes/e5_flat.index
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.retrieval.e5_retriever import build_faiss_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--model_name", default="intfloat/e5-base-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--index_type", choices=["flat", "hnsw"], default="flat")
    args = parser.parse_args()
    build_faiss_index(args.corpus_path, args.index_path, args.model_name, args.batch_size,
                      index_type=args.index_type)
    print(f"Index written to {args.index_path}")


if __name__ == "__main__":
    main()
