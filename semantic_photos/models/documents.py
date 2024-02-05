from typing import List, Dict, Optional, Any
import calendar

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document

from .schema import ImageData
from . import HUGGINGFACE_CACHE


class ImageDocument:

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
            **(model_kwargs or {})
        )
        self.db = Chroma(
            persist_directory=chroma_persist_path,
            collection_name=collection_name,
            embedding_function=self.model
        )

    def add_images(self, images: List[ImageData]):
        documents = []

        for img in images:
            documents.append(Document(
                page_content=img.text,
                metadata={
                    "path": img.path,
                    "album": img.album_name,
                    "name": img.file_name,
                    "year": img.created.year,
                    "month": calendar.month_name[img.created.month],
                    "day": calendar.day_name[img.created.weekday()],
                    "@date": img.created.date().strftime('%Y-%m-%d'),
                    "@timestamp": img.created.timestamp()
                }
            ))

        self.db.add_documents(documents=documents)

    def query(self, query: str):
        hits = self.db.similarity_search_with_score(
            query=query,
            k=10
        )
        return hits

    def teardown(self):
        self.db.delete_collection()
