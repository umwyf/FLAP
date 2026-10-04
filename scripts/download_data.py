#!/usr/bin/env python
"""Download the FlashRAG-formatted QA datasets and (optionally) the wiki-18 corpus.

Example::

    python scripts/download_data.py --data_root data --datasets all --corpus
"""

import argparse
import os
import subprocess
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flap.data.datasets import DATASETS  # noqa: E402

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"      # wiki-18.jsonl.gz (Search-R1 release)
INDEX_REPO = "PeterJinGo/wiki-18-e5-index"     # prebuilt E5 flat index (optional, large)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--corpus", action="store_true", help="also download the wiki-18 corpus")
    parser.add_argument("--index", action="store_true", help="also download the prebuilt E5 FAISS index")
    args = parser.parse_args()

    names = list(DATASETS) if args.datasets == ["all"] else args.datasets
    patterns = [f"{DATASETS[n]['dir']}/*" for n in names]
    print(f"Downloading {names} from {FLASHRAG_REPO} ...")
    snapshot_download(repo_id=FLASHRAG_REPO, repo_type="dataset", local_dir=args.data_root,
                      allow_patterns=patterns)

    if args.corpus:
        print(f"Downloading wiki-18 corpus from {CORPUS_REPO} ...")
        snapshot_download(repo_id=CORPUS_REPO, repo_type="dataset", local_dir=args.data_root,
                          allow_patterns=["wiki-18.jsonl.gz", "wiki-18.jsonl"])
        gz = os.path.join(args.data_root, "wiki-18.jsonl.gz")
        if os.path.exists(gz) and not os.path.exists(gz[:-3]):
            print("Decompressing corpus ...")
            subprocess.run(["gzip", "-dk", gz], check=True)

    if args.index:
        index_dir = os.path.join(args.data_root, "..", "indexes")
        print(f"Downloading prebuilt E5 index from {INDEX_REPO} into {index_dir} ...")
        snapshot_download(repo_id=INDEX_REPO, repo_type="dataset", local_dir=index_dir)
        print("If the index is split into parts, concatenate them, e.g. `cat part_* > e5_Flat.index`.")
    print("Done.")


if __name__ == "__main__":
    main()
