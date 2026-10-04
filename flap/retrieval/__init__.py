"""Retrieval backends.

* :class:`E5Retriever` – local dense retrieval with E5 embeddings and a FAISS
  index over the 2018 Wikipedia dump (the setting used in the paper).
* :class:`HTTPRetriever` – client for a retrieval server exposing the
  Search-R1 ``/retrieve`` API (``flap.retrieval.server``).
* :class:`InMemoryRetriever` – tiny lexical retriever for tests and demos.
"""

from flap.retrieval.base import Document, Retriever, InMemoryRetriever, build_retriever
from flap.retrieval.client import HTTPRetriever

__all__ = ["Document", "Retriever", "InMemoryRetriever", "HTTPRetriever", "build_retriever"]
