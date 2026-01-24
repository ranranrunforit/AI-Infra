"""
Vector Database Indexing Module

This module handles the indexing of documents into vector databases (Pinecone, ChromaDB).
It efficiently manages embedding generation, batching, and metadata storage.

Learning Objectives:
1. Understand vector database indexing operations
2. Implement batch processing for efficiency
3. Handle different vector store APIs (Pinecone vs Chroma)
4. Manage document metadata and IDs
5. Implement error handling and retries
"""

import logging
import uuid
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import pinecone
except ImportError:
    pinecone = None

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocument:
    """
    Represents a document ready for indexing.
    
    Attributes:
        id: Unique identifier
        text: Text content
        metadata: Associated metadata (source, page, etc)
        embedding: Vector embedding (optional, computed if None)
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None


class VectorIndexer(ABC):
    """
    Abstract base class for vector indexers.
    """

    def __init__(self, embedding_model=None, batch_size: int = 100):
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    async def index_documents(self, documents: List[IndexedDocument]) -> int:
        """
        Index a list of documents.
        Handles embedding generation and batch upserts.
        """
        total_indexed = 0
        
        # 1. Generate embeddings if needed
        docs_to_embed = [doc for doc in documents if doc.embedding is None]
        if docs_to_embed and self.embedding_model:
            texts = [doc.text for doc in docs_to_embed]
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            try:
                # Handle different embedding model interfaces
                if callable(self.embedding_model):
                    embeddings = self.embedding_model(texts)
                elif hasattr(self.embedding_model, "embed_documents"):
                    embeddings = self.embedding_model.embed_documents(texts)
                else:
                    logger.warning("Embedding model interface unknown, skipping embedding generation.")
                    embeddings = []

                if embeddings and len(embeddings) == len(docs_to_embed):
                    for doc, emb in zip(docs_to_embed, embeddings):
                        doc.embedding = emb
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                # We might continue with docs that have embeddings or fail?
                # For now, let's allow it to fail or skip
                pass

        # 2. Batch upsert
        # Filter out docs that still (or originally) don't have embeddings if indexer requires them
        valid_docs = [doc for doc in documents if doc.embedding is not None]
        if len(valid_docs) < len(documents):
            logger.warning(f"Skipping {len(documents) - len(valid_docs)} documents without embeddings.")

        for i in range(0, len(valid_docs), self.batch_size):
            batch = valid_docs[i : i + self.batch_size]
            try:
                await self._upsert_batch(batch)
                total_indexed += len(batch)
                logger.debug(f"Indexed batch {i//self.batch_size + 1}: {len(batch)} documents")
            except Exception as e:
                logger.error(f"Failed to index batch {i}: {e}")
                raise e
                
        return total_indexed

    @abstractmethod
    async def _upsert_batch(self, documents: List[IndexedDocument]):
        """Upsert a batch of documents."""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]):
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    async def delete_collection(self):
        """Delete the entire collection/index."""
        pass


class PineconeIndexer(VectorIndexer):
    """
    Indexer for Pinecone vector database.
    """
    def __init__(
        self, 
        api_key: str, 
        index_name: str, 
        environment: str = "us-west1-gcp", 
        embedding_model=None,
        dimension: int = 768
    ):
        super().__init__(embedding_model)
        if pinecone is None:
             raise ImportError("Pinecone client not installed. Run `pip install pinecone-client`.")
        
        self.index_name = index_name
        
        try:
            # Modern Pinecone client
            self.pc = pinecone.Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
        except AttributeError:
             # Legacy Pinecone client
             pinecone.init(api_key=api_key, environment=environment)
             self.index = pinecone.Index(index_name)

    async def _upsert_batch(self, documents: List[IndexedDocument]):
        vectors = []
        for doc in documents:
            if not doc.embedding:
                continue
            
            # Pinecone metadata must be primitives
            safe_metadata = {
                k: v for k, v in doc.metadata.items() 
                if isinstance(v, (str, int, float, bool, list))
            }
            safe_metadata["text"] = doc.text
            
            vectors.append({
                "id": doc.id,
                "values": doc.embedding,
                "metadata": safe_metadata
            })
            
        if vectors:
            self.index.upsert(vectors=vectors)

    async def delete_documents(self, document_ids: List[str]):
        if document_ids:
            self.index.delete(ids=document_ids)

    async def delete_collection(self):
        try:
            self.index.delete(delete_all=True)
        except Exception as e:
            logger.error(f"Error deleting all from pinecone: {e}")


class ChromaIndexer(VectorIndexer):
    """
    Indexer for ChromaDB.
    """
    def __init__(
        self, 
        collection_name: str, 
        persist_directory: Optional[str] = None,
        embedding_model=None
    ):
        super().__init__(embedding_model)
        if chromadb is None:
            raise ImportError("ChromaDB not installed. Run `pip install chromadb`.")

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
            
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def _upsert_batch(self, documents: List[IndexedDocument]):
        if not documents:
            return

        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    async def delete_documents(self, document_ids: List[str]):
        if document_ids:
            self.collection.delete(ids=document_ids)
            
    async def delete_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
        except ValueError:
            pass


def create_indexer(config: Dict[str, Any], embedding_model=None) -> VectorIndexer:
    """
    Factory to create appropriate indexer.
    """
    idx_type = config.get("type", "chroma").lower()
    
    if idx_type == "pinecone":
        return PineconeIndexer(
            api_key=config.get("api_key", ""),
            environment=config.get("environment", ""),
            index_name=config.get("index_name", "default"),
            embedding_model=embedding_model
        )
    elif idx_type == "chroma":
        return ChromaIndexer(
            collection_name=config.get("collection_name", "default"),
            persist_directory=config.get("persist_directory", "./chroma_db"),
            embedding_model=embedding_model
        )
    else:
        raise ValueError(f"Unknown indexer type: {idx_type}")
