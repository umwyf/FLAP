"""Dense retrieval with E5 (Wang et al., 2024) over a FAISS index.

Corpus format (FlashRAG / Search-R1 ``wiki-18.jsonl``): one JSON object per
line with fields ``id`` and ``contents`` where ``contents`` is
``"title"\\ntext``.  Build the index with ``scripts/build_index.py``.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np
import torch

from flap.retrieval.base import Document, Retriever


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class E5Encoder:
    """E5 encoder with the ``query: `` / ``passage: `` prefixes and mean pooling."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: Optional[str] = None,
                 max_length: int = 256, fp16: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        if fp16 and self.device.startswith("cuda"):
            self.model = self.model.half()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: Sequence[str], prefix: str, batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, max_length=self.max_length, padding=True, truncation=True,
                                 return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            emb = torch.nn.functional.normalize(_mean_pool(hidden, enc["attention_mask"]), dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

    def encode_queries(self, queries: Sequence[str], batch_size: int = 128) -> np.ndarray:
        return self.encode(queries, "query: ", batch_size)

    def encode_passages(self, passages: Sequence[str], batch_size: int = 128) -> np.ndarray:
        return self.encode(passages, "passage: ", batch_size)


def load_corpus(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Document.from_contents(obj.get("id", len(docs)), obj["contents"]))
    return docs


class E5Retriever(Retriever):
    def __init__(self, index_path: str, corpus_path: str, model_name: str = "intfloat/e5-base-v2",
                 device: Optional[str] = None, use_gpu_index: bool = False):
        import faiss

        self.encoder = E5Encoder(model_name, device=device)
        self.index = faiss.read_index(index_path)
        if use_gpu_index and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.corpus = load_corpus(corpus_path)
        if self.index.ntotal != len(self.corpus):
            raise ValueError(
                f"Index size ({self.index.ntotal}) does not match corpus size ({len(self.corpus)})."
            )

    def batch_search(self, queries: Sequence[str], top_k: int = 3) -> List[List[Document]]:
        if not queries:
            return []
        q = self.encoder.encode_queries(list(queries))
        scores, ids = self.index.search(q.astype(np.float32), top_k)
        results = []
        for row_scores, row_ids in zip(scores, ids):
            docs = []
            for s, i in zip(row_scores, row_ids):
                if i < 0:
                    continue
                d = self.corpus[int(i)]
                docs.append(Document(d.id, d.title, d.text, score=float(s)))
            results.append(docs)
        return results


def build_faiss_index(corpus_path: str, index_path: str, model_name: str = "intfloat/e5-base-v2",
                      batch_size: int = 256, device: Optional[str] = None, index_type: str = "flat"):
    """Encode a corpus and write a FAISS inner-product index to ``index_path``."""
    import faiss
    from tqdm import tqdm

    encoder = E5Encoder(model_name, device=device)
    docs = load_corpus(corpus_path)
    dim = encoder.model.config.hidden_size
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    for i in tqdm(range(0, len(docs), batch_size), desc="Encoding corpus"):
        chunk = docs[i : i + batch_size]
        emb = encoder.encode_passages([d.contents for d in chunk], batch_size=batch_size)
        index.add(emb.astype(np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    faiss.write_index(index, index_path)
    return index
