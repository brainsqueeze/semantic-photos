from typing import Any
import calendar

from sentence_transformers import SentenceTransformer
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from chromadb.api.types import Metadata, SearchResultRow
from chromadb.api.models.Collection import Collection
import chromadb

from .schema import ImageData
from . import HUGGINGFACE_CACHE


def _rrf_score(score: float, rank: int, k: float = 60.) -> float:
    return score + (1. / (k + rank))


def rrf_rerank(*result_sets: chromadb.QueryResult, k: float = 60., n_results: int = 10) -> list[SearchResultRow]:
    rescored = {}
    documents = {}
    keys = ("ids", "documents", "metadatas")

    for results in result_sets:
        docs = [
            dict(zip(keys, b))
            for b in zip(*[(results[k] or [])[0] for k in keys])
        ]

        for result_rank, doc in enumerate(docs, start=1):
            doc_id = doc["ids"]
            if doc_id not in documents:
                documents[doc_id] = doc

            rescored[doc_id] = _rrf_score(score=rescored.get(doc_id) or 0., k=k, rank=result_rank)

    output = []
    for doc_id, score in sorted(rescored.items(), key=lambda _: _[1], reverse=True)[:n_results]:
        output.append(SearchResultRow(
            id=doc_id,
            metadata=documents[doc_id]["metadatas"],
            document=documents[doc_id]["documents"],
            score=score
        ))

    return output


class TfIdfSearch:

    def __init__(self, collection: Collection):
        self._collection = collection
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 25), dtype='float32')

        batch_size = 128
        _docs = []
        for i in range(0, collection.count(), batch_size):
            batch = collection.get(include=["documents", "metadatas"], limit=batch_size, offset=i)
            if batch["documents"] is None:
                continue

            _docs.extend(batch["documents"])
        self._index = self.vectorizer.fit_transform(_docs).T # pyright: ignore[reportAttributeAccessIssue]

    def _inner(self, query: str, top_k: int = 10):
        x_ = self.vectorizer.transform([query])
        retrieved = x_ @ self._index

        row = retrieved.tocsr()[0] # get the single sparse row
        if row.nnz == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        order = np.argsort(-row.data)[:top_k]  # sort descending
        top_idx = row.indices[order]
        top_scores = row.data[order]

        return top_idx, top_scores

    def fts(self, query: str, n_results: int = 10) -> chromadb.QueryResult:
        indices, scores = self._inner(query, top_k=n_results)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[Metadata] = []
        distances: list[float] = []
        for idx, score in zip(indices, scores):
            doc = self._collection.get(limit=1, offset=idx)

            ids.extend(doc["ids"])
            documents.extend(doc["documents"] or [])
            metadatas.extend(doc["metadatas"] or [])
            distances.append(score.item())

        return chromadb.QueryResult(
            ids=[ids],
            metadatas=[metadatas],
            documents=[documents],
            distances=[distances]
        ) # type: ignore


class SentenceTransformersEmbedding(chromadb.EmbeddingFunction):

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        cache_folder: str = HUGGINGFACE_CACHE,
        device: str | torch.device = "cpu",
        model_kwargs: dict[str, Any] | None = None
    ) -> None:
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_folder,
            device=str(device),
            model_kwargs=(model_kwargs or {})
        )
        super().__init__()

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self.model.encode(sentences=input).tolist()



class ImageVectorStore:
    """ChromaDB vector store wrapper for image search

    Parameters
    ----------
    chroma_persist_path : str
        Path containing (or to contain) the ChromaDB SQLite file(s)
    collection_name : str, optional
        ChromaDB collection, by default "semantic-photos"
    model_name : str, optional
        Text embedding model, by default "sentence-transformers/all-MiniLM-L12-v2"
    cache_folder : str, optional
        Folder containing model cache files, by default HUGGINGFACE_CACHE
    model_kwargs : Optional[dict[str, Any]], optional
        Optional kwargs for Huggingface model inference, by default None
    """

    def __init__(
        self,
        chroma_persist_path: str,
        collection_name: str = "semantic-photos",
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        cache_folder: str = HUGGINGFACE_CACHE,
        device: str | torch.device = "cpu",
        model_kwargs: dict[str, Any] | None = None,
        build_tfidf_index: bool = False
    ):
        if isinstance(device, torch.device):
            device = str(device)

        chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
        self.db = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformersEmbedding(
                model_name=model_name,
                cache_folder=cache_folder,
                device=device,
                model_kwargs=model_kwargs
            ),
            configuration={
                "hnsw": {"space": "ip"}
            }
        )

        self.ft_index = None
        if build_tfidf_index:
            self.ft_index = TfIdfSearch(self.db)

    def add_images(self, images: list[ImageData]) -> None:
        """Index a batch of images and metadata

        Parameters
        ----------
        images : list[ImageData]
        """

        ids = []
        texts = []
        metadatas = []

        for img in images:
            ids.append(img.path)
            texts.append(img.text)
            metadatas.append({
                "path": img.path,
                "album": img.album_name,
                "name": img.file_name,
                "year": img.created.year,
                "month": calendar.month_name[img.created.month],
                "day": calendar.day_name[img.created.weekday()],
                "caption": img.caption,
                "people_description": img.people_description,
                "location_description": img.geo_description,
                "@date": img.created.date().strftime('%Y-%m-%d'),
                "@timestamp": img.created.timestamp()
            })
        self.db.add(ids=ids, documents=texts, metadatas=metadatas)

    def search(
        self,
        query: str,
        n_results: int = 10,
        where_document: chromadb.WhereDocument | None = None
    ) -> list[SearchResultRow]:
        """Run a vector search query to return documents and search scores.

        Parameters
        ----------
        query : str
            Text prompt to retrieve documents
        n_results : int, optional
            How many images to return, by default 10
        where_document : chromadb.WhereDocument | None, optional
            A WhereDocument type dict used to filter by the documents.
            E.g. `{$contains: {"text": "hello"}}`, by default None

        Returns
        -------
        list[tuple[Document, float]]
            (Image document, score)
        """

        retriever_results = [
            self.knn(query=query, n_results=100, where_document=where_document),
        ]

        if self.ft_index is not None:
            retriever_results.append(self.ft_index.fts(query=query, n_results=100))

        return rrf_rerank(*retriever_results, n_results=n_results, k=10)

    def knn(
        self,
        query: str,
        n_results: int = 10,
        where_document: chromadb.WhereDocument | None = None
    ) -> chromadb.QueryResult:
        """Run a vector search query to return documents and search scores.

        Parameters
        ----------
        query : str
            Text prompt to retrieve documents
        n_results : int, optional
            How many images to return, by default 10
        where_document : dict[str, str] | None, optional
            A WhereDocument type dict used to filter by the documents.
            E.g. `{$contains: {"text": "hello"}}`, by default None

        Returns
        -------
        list[tuple[Document, float]]
            (Image document, score)
        """

        return self.db.query(
            query_texts=query,
            n_results=n_results,
            where_document=where_document
        )

    def __len__(self) -> int:
        return self.db.count()
