from typing import Any
from dataclasses import dataclass
import calendar

from sentence_transformers import SentenceTransformer

from chromadb.execution.expression.operator import Rank, Key
from chromadb.api.types import SearchResult
import chromadb

from .schema import ImageData
from . import HUGGINGFACE_CACHE


class SentenceTransformersEmbedding(chromadb.EmbeddingFunction):

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        cache_folder: str = HUGGINGFACE_CACHE,
        model_kwargs: dict[str, Any] | None = None
    ) -> None:
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_folder,
            model_kwargs=(model_kwargs or {})
        )
        super().__init__()

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self.model.encode(sentences=input).tolist()


@dataclass
class FullTextSearch(Rank):
    query: str
    key: str | Key = chromadb.K.DOCUMENT
    limit: int = 16
    return_rank: bool = False

    def to_dict(self) -> dict[str, Any]:
        key_value = self.key
        if isinstance(key_value, Key):
            key_value = key_value.name

        # Build result dict - only include non-default values to keep JSON clean
        result = {"query": self.query, "key": key_value, "limit": self.limit}
        if self.return_rank:  # Only include if True (non-default)
            result["return_rank"] = self.return_rank

        return {"$contains": result}


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
        model_kwargs: dict[str, Any] | None = None,
    ):
        chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
        self.db = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformersEmbedding(
                model_name=model_name,
                cache_folder=cache_folder,
                model_kwargs=model_kwargs
            ),
            configuration={
                "hnsw": {"space": "ip"}
            }
        )

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
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None
    ) -> SearchResult:
        """Run a vector search query to return documents and search scores.

        Parameters
        ----------
        query : str
            Text prompt to retrieve documents
        n_results : int, optional
            How many images to return, by default 10
        where : dict[str, str] | None, optional
            Where filter, equivalent to `where` in the ChromaDB API, or `filter` in LangChain, by default None
        where_document : dict[str, str] | None, optional
            A WhereDocument type dict used to filter by the documents.
            E.g. `{$contains: {"text": "hello"}}`, by default None

        Returns
        -------
        list[tuple[Document, float]]
            (Image document, score)
        """

        rrf_retrievers = chromadb.Rrf(
            ranks=[
                chromadb.Knn(query=query, return_rank=True, limit=100),
                FullTextSearch(query=query, return_rank=True, limit=100)
            ]
        )

        search = chromadb.Search().where(where).rank(rrf_retrievers).limit(n_results)
        return self.db.search(search)

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
