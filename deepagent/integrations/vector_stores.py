"""
Vector Store Integrations

Provides persistent storage for episodic memory using vector databases:
- Chroma (default, simple to use)
- Qdrant (production-grade, scalable)

Author: Oluwafemi Idiakhoa
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class VectorEntry:
    """Entry in vector store"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Search result from vector store"""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float


class VectorStore(ABC):
    """Base class for vector stores"""

    @abstractmethod
    def add(self, entries: List[VectorEntry]) -> List[str]:
        """Add entries to vector store"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete entries by ID"""
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[VectorEntry]:
        """Get entry by ID"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all entries"""
        pass


class ChromaVectorStore(VectorStore):
    """
    Chroma vector store implementation

    Simple, lightweight, perfect for development and small-scale production.
    """

    def __init__(self, collection_name: str = "episodic_memory", persist_directory: str = "./chroma_db"):
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "DeepAgent episodic memory storage"}
            )

        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

    def add(self, entries: List[VectorEntry]) -> List[str]:
        """Add entries to Chroma"""
        if not entries:
            return []

        ids = [entry.id for entry in entries]
        documents = [entry.content for entry in entries]
        embeddings = [entry.embedding.tolist() for entry in entries]
        metadatas = [entry.metadata for entry in entries]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return ids

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """Search Chroma for similar vectors"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity=1.0 - results['distances'][0][i]  # Convert distance to similarity
            ))

        return search_results

    def delete(self, ids: List[str]) -> bool:
        """Delete entries from Chroma"""
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception:
            return False

    def get(self, id: str) -> Optional[VectorEntry]:
        """Get entry by ID from Chroma"""
        try:
            result = self.collection.get(ids=[id], include=["documents", "embeddings", "metadatas"])

            if not result['ids']:
                return None

            return VectorEntry(
                id=result['ids'][0],
                content=result['documents'][0],
                embedding=np.array(result['embeddings'][0]),
                metadata=result['metadatas'][0]
            )
        except Exception:
            return None

    def clear(self) -> bool:
        """Clear all entries from Chroma"""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description": "DeepAgent episodic memory storage"}
            )
            return True
        except Exception:
            return False


class QdrantVectorStore(VectorStore):
    """
    Qdrant vector store implementation

    Production-grade, scalable, supports filtering and advanced features.
    """

    def __init__(
        self,
        collection_name: str = "episodic_memory",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if url:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                self.client = QdrantClient(host=host, port=port)

            self.collection_name = collection_name

            # Create collection if it doesn't exist
            collections = [col.name for col in self.client.get_collections().collections]
            if collection_name not in collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )

        except ImportError:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )

    def add(self, entries: List[VectorEntry]) -> List[str]:
        """Add entries to Qdrant"""
        if not entries:
            return []

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=entry.id,
                vector=entry.embedding.tolist(),
                payload={"content": entry.content, **entry.metadata}
            )
            for entry in entries
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return [entry.id for entry in entries]

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """Search Qdrant for similar vectors"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        search_results = []
        for hit in results:
            payload = hit.payload
            content = payload.pop("content", "")
            search_results.append(SearchResult(
                id=str(hit.id),
                content=content,
                metadata=payload,
                similarity=hit.score
            ))

        return search_results

    def delete(self, ids: List[str]) -> bool:
        """Delete entries from Qdrant"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            return True
        except Exception:
            return False

    def get(self, id: str) -> Optional[VectorEntry]:
        """Get entry by ID from Qdrant"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_vectors=True
            )

            if not result:
                return None

            point = result[0]
            payload = point.payload
            content = payload.pop("content", "")

            return VectorEntry(
                id=str(point.id),
                content=content,
                embedding=np.array(point.vector),
                metadata=payload
            )
        except Exception:
            return None

    def clear(self) -> bool:
        """Clear all entries from Qdrant"""
        try:
            self.client.delete_collection(self.collection_name)

            from qdrant_client.models import Distance, VectorParams
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            return True
        except Exception:
            return False
