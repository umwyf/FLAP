from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class Document:
    id: str
    title: str
    text: str
    score: Optional[float] = None

    @property
    def contents(self) -> str:
        """FlashRAG-style ``"title"\\ntext`` rendering."""
        return f'"{self.title}"\n{self.text}'

    @classmethod
    def from_contents(cls, doc_id: str, contents: str, score: Optional[float] = None) -> "Document":
        """Parse a FlashRAG-style ``contents`` field (first line is the quoted title)."""
        lines = contents.split("\n")
        title = lines[0].strip().strip('"') if lines else ""
        text = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        return cls(id=str(doc_id), title=title, text=text, score=score)

    def to_dict(self) -> Dict:
        return {"id": self.id, "title": self.title, "text": self.text, "score": self.score}


class Retriever(ABC):
    @abstractmethod
    def batch_search(self, queries: Sequence[str], top_k: int = 3) -> List[List[Document]]:
        ...

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        return self.batch_search([query], top_k=top_k)[0]


_TOKEN_RE = re.compile(r"\w+")


class InMemoryRetriever(Retriever):
    """A tiny bag-of-words retriever for unit tests, demos and smoke tests."""

    def __init__(self, documents: Sequence[Document]):
        self.documents = list(documents)
        self._tokens = [set(_TOKEN_RE.findall((d.title + " " + d.text).lower())) for d in self.documents]

    def batch_search(self, queries: Sequence[str], top_k: int = 3) -> List[List[Document]]:
        results = []
        for q in queries:
            q_tokens = set(_TOKEN_RE.findall(q.lower()))
            scored = []
            for doc, toks in zip(self.documents, self._tokens):
                overlap = len(q_tokens & toks)
                if overlap:
                    scored.append((overlap / (len(q_tokens) + 1e-9), doc))
            scored.sort(key=lambda x: -x[0])
            results.append(
                [Document(d.id, d.title, d.text, score=s) for s, d in scored[:top_k]]
            )
        return results


def build_retriever(kind: str, **kwargs) -> Retriever:
    """Factory: ``kind`` in {"http", "e5"}."""
    kind = kind.lower()
    if kind == "http":
        from flap.retrieval.client import HTTPRetriever

        return HTTPRetriever(**kwargs)
    if kind == "e5":
        from flap.retrieval.e5_retriever import E5Retriever

        return E5Retriever(**kwargs)
    raise ValueError(f"Unknown retriever kind: {kind}")
