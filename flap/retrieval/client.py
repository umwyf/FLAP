"""HTTP client for a Search-R1-compatible retrieval server.

Request:  ``POST {url}/retrieve`` with ``{"queries": [...], "topk": k, "return_scores": true}``
Response: ``{"result": [[{"document": {"id": ..., "contents": ...}, "score": ...}, ...], ...]}``
"""

from __future__ import annotations

import time
from typing import List, Sequence

import requests

from flap.retrieval.base import Document, Retriever


class HTTPRetriever(Retriever):
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 120.0, max_retries: int = 5):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def batch_search(self, queries: Sequence[str], top_k: int = 3) -> List[List[Document]]:
        if not queries:
            return []
        payload = {"queries": list(queries), "topk": top_k, "return_scores": True}
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(f"{self.url}/retrieve", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        results: List[List[Document]] = []
        for hits in data["result"]:
            docs = []
            for h in hits:
                if isinstance(h, dict) and "document" in h:
                    doc, score = h["document"], h.get("score")
                else:
                    doc, score = h, None
                docs.append(Document.from_contents(doc.get("id", ""), doc.get("contents", ""), score))
            results.append(docs)
        return results
