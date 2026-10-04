"""Retrieval server exposing the Search-R1 ``/retrieve`` API.

Run with::

    python -m flap.retrieval.server --index_path indexes/e5_flat.index \\
        --corpus_path data/wiki-18.jsonl --port 8000
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from flap.retrieval.e5_retriever import E5Retriever


class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 3
    return_scores: bool = True


def create_app(retriever: E5Retriever) -> FastAPI:
    app = FastAPI(title="FLAP retrieval server")

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        results = retriever.batch_search(req.queries, top_k=req.topk or 3)
        payload = []
        for hits in results:
            row = []
            for d in hits:
                item = {"document": {"id": d.id, "contents": d.contents}}
                if req.return_scores:
                    item["score"] = d.score
                row.append(item)
            payload.append(row)
        return {"result": payload}

    @app.get("/health")
    def health():
        return {"status": "ok", "num_docs": len(retriever.corpus)}

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--model_name", default="intfloat/e5-base-v2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu_index", action="store_true")
    args = parser.parse_args()

    retriever = E5Retriever(args.index_path, args.corpus_path, args.model_name, use_gpu_index=args.gpu_index)
    uvicorn.run(create_app(retriever), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
