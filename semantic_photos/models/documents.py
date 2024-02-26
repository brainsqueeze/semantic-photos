from typing import List, Dict, Tuple, Optional, Any
import calendar

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma

from .schema import ImageData
from . import HUGGINGFACE_CACHE


class ImageVectorStore:
    """ChromaDB vector store wrapper for image search

    Parameters
    ----------
    chroma_persist_path : str
        Path containing (or to contain) the ChromaDB SQLite file(s)
    collection_name : str, optional
        ChromaDB collection, by default "semantic-photos"
    model_name : str, optional
        Text embedding model, by default "sentence-transformers/all-mpnet-base-v2"
    cache_folder : str, optional
        Folder containing model cache files, by default HUGGINGFACE_CACHE
    model_kwargs : Optional[Dict[str, Any]], optional
        Optional kwargs for Huggingface model inference, by default None
    """

    def __init__(
        self,
        chroma_persist_path: str,
        collection_name: str = "semantic-photos",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_folder: str = HUGGINGFACE_CACHE,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs
        )
        self.db = Chroma(
            persist_directory=chroma_persist_path,
            collection_name=collection_name,
            embedding_function=self.model
        )

    def add_images(self, images: List[ImageData]) -> None:
        """Index a batch of images and metadata

        Parameters
        ----------
        images : List[ImageData]
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
        self.db.add_texts(ids=ids, texts=texts, metadatas=metadatas)

    def query(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """Run a vector search query to return documents and search scores.

        Parameters
        ----------
        query : str
            Text prompt to retrieve documents
        n_results : int, optional
            How many images to return, by default 10
        where : Optional[Dict[str, str]], optional
            Where filter, equivalent to `where` in the ChromaDB API, or `filter` in LangChain, by default None
        where_document : Optional[Dict[str, str]], optional
            A WhereDocument type dict used to filter by the documents.
            E.g. `{$contains: {"text": "hello"}}`, by default None

        Returns
        -------
        List[Tuple[Document, float]]
            (Image document, score)
        """

        hits = self.db.similarity_search_with_score(
            query=query,
            k=n_results,
            filter=where,
            where_document=where_document
        )
        return hits

    def teardown(self):
        """Delete ChromaDB collection, only use when you want to remove the database, mainly for testing purposes.
        """

        self.db.delete_collection()

    def __len__(self) -> int:
        return self.db._collection.count()
