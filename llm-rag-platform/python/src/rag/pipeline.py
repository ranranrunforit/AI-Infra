"""
RAG Pipeline Implementation for Enterprise LLM Platform
Supports two-stage retrieval: vector search → reranking → LLM generation

Original: Uses BAAI/bge-large-en-v1.5 for embeddings, vLLM for generation
GCP/Colab Adaptation: Same HuggingFace embeddings, Gemini Pro for generation

Works on Google Colab (free tier with bge-small, Pro with bge-large).
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with metadata"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    # Embedding model (HuggingFace sentence-transformers - same as original)
    # Use bge-small for Colab free tier (384 dims, <500MB)
    # Use bge-large for better quality with more RAM (1024 dims, ~1.5GB)
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # Colab-friendly default
    embedding_batch_size: int = 32

    # Vector database
    vector_db_host: str = "localhost"
    vector_db_port: int = 6333
    collection_name: str = "enterprise_knowledge"

    # Retrieval parameters (same as original architecture)
    retrieval_top_k: int = 100  # Initial retrieval
    rerank_top_k: int = 10     # After reranking
    min_relevance_score: float = 0.7

    # LLM parameters - routed through LLMGateway
    # Set LLM_BACKEND=gemini to use Gemini Pro
    # Set LLM_BACKEND=vllm to use self-hosted vLLM
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_top_p: float = 0.95

    # Context window
    max_context_tokens: int = 3000  # Reserve tokens for context

    # Reranking (cross-encoder, same as original)
    rerank_model: str = "BAAI/bge-reranker-base"  # Lighter than bge-reranker-large
    enable_reranking: bool = True

    # Safety
    enable_guardrails: bool = True

    # LLM backend
    llm_backend: str = "gemini"  # gemini | vllm | openai
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    vllm_endpoint: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create config from environment variables (Colab / Cloud Run friendly)"""
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            vector_db_host=os.getenv("QDRANT_HOST", "localhost"),
            vector_db_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("COLLECTION_NAME", "enterprise_knowledge"),
            llm_backend=os.getenv("LLM_BACKEND", "gemini"),
            gemini_api_key=os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            vllm_endpoint=os.getenv("VLLM_ENDPOINT"),
            enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        )


class RAGPipeline:
    """
    Two-stage RAG pipeline (same architecture as original):
    1. Dense retrieval: semantic search using vector embeddings (HuggingFace bge)
    2. Reranking: cross-encoder to rerank top candidates (bge-reranker)
    3. Generation: LLM with retrieved context (Gemini Pro or vLLM)

    Designed to work on:
    - Google Colab free tier (bge-small embeddings, Gemini API via GOOGLE_API_KEY)
    - GCP Cloud Run (same, containerised)
    - GCP VM with GPU (switch EMBEDDING_MODEL to bge-large, LLM_BACKEND to vllm)
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        # Initialize embedding model (HuggingFace sentence-transformers - unchanged from original)
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded: dim={self.embedding_dim}")

        # Initialize reranking model (cross-encoder - same as original)
        if config.enable_reranking:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranking model: {config.rerank_model}")
            self.rerank_model = CrossEncoder(config.rerank_model)

        # Initialize vector database client (Qdrant - unchanged from original)
        self.vector_db = QdrantClient(
            host=config.vector_db_host,
            port=config.vector_db_port,
            timeout=30.0
        )

        # Initialize collection if not exists
        self._init_collection()

        # Initialize LLM Gateway (Gemini Pro or vLLM)
        from src.llm.gateway import LLMGateway
        self.llm_gateway = LLMGateway(
            backend=config.llm_backend,
            gemini_api_key=config.gemini_api_key,
            gemini_model=config.gemini_model,
            vllm_endpoint=config.vllm_endpoint,
        )

        logger.info(
            f"RAG pipeline initialized | "
            f"embedding={config.embedding_model} | "
            f"llm_backend={config.llm_backend} | "
            f"available_backends={self.llm_gateway.available_backends}"
        )

    def _init_collection(self):
        """Initialize Qdrant collection for vector storage"""
        collections = self.vector_db.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.config.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.config.collection_name}")
            self.vector_db.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info("Collection created successfully")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch of texts"""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings

    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """Add documents to vector database"""
        logger.info(f"Adding {len(documents)} documents to vector database")

        # Generate embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.embed_batch(texts)

        # Prepare points for Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=doc.id,
                vector=embedding.tolist(),
                payload={
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "indexed_at": datetime.utcnow().isoformat()
                }
            )
            points.append(point)

        # Batch upsert
        total_uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.vector_db.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )
            total_uploaded += len(batch)
            logger.info(f"Uploaded {total_uploaded}/{len(points)} documents")

        return total_uploaded

    async def retrieve_dense(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Stage 1: Dense retrieval using vector similarity
        Returns top-k most similar documents
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k

        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        # Search vector database
        results = self.vector_db.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=self.config.min_relevance_score
        )

        # Convert to Document objects
        documents = []
        for result in results:
            doc = Document(
                id=result.id,
                text=result.payload["text"],
                metadata=result.payload["metadata"],
                score=result.score
            )
            documents.append(doc)

        logger.info(f"Dense retrieval found {len(documents)} documents (query: {query[:50]}...)")
        return documents

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        Stage 2: Rerank documents using cross-encoder
        More accurate but slower than vector search
        """
        if not self.config.enable_reranking or not documents:
            return documents[:top_k or self.config.rerank_top_k]

        if top_k is None:
            top_k = self.config.rerank_top_k

        # Prepare query-document pairs for cross-encoder
        pairs = [[query, doc.text] for doc in documents]

        # Compute reranking scores (cross-encoder gives relevance scores directly)
        rerank_scores = await asyncio.to_thread(
            self.rerank_model.predict,
            pairs,
        )

        # Update document scores
        for doc, score in zip(documents, rerank_scores):
            doc.score = float(score)

        # Sort by rerank score
        documents.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Reranking selected top {top_k} documents")
        return documents[:top_k]

    def _build_context(
        self,
        documents: List[Document],
        max_tokens: int = None
    ) -> str:
        """Build context string from documents, respecting token limit"""
        if max_tokens is None:
            max_tokens = self.config.max_context_tokens

        context_parts = []
        total_tokens = 0

        for i, doc in enumerate(documents):
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            doc_tokens = len(doc.text) // 4

            if total_tokens + doc_tokens > max_tokens:
                logger.info(f"Context limit reached, using {i} of {len(documents)} documents")
                break

            # Format document with metadata
            source = doc.metadata.get("source", "Unknown")
            timestamp = doc.metadata.get("timestamp", "N/A")

            context_parts.append(
                f"[Document {i+1}] (Source: {source}, Date: {timestamp})\n"
                f"{doc.text}\n"
            )
            total_tokens += doc_tokens

        context = "\n".join(context_parts)
        logger.info(f"Built context with {len(context_parts)} documents (~{total_tokens} tokens)")
        return context

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        contains_pii: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage 3: Generate response using LLM Gateway (Gemini Pro or vLLM)
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant. Answer the question based on the provided context. "
                "If the context doesn't contain enough information, say so. "
                "Always cite the document numbers when referencing information."
            )

        # Build prompt (same format as original)
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Call LLM via Gateway (Gemini or vLLM)
        logger.info(f"Generating response via {self.config.llm_backend}")
        response = await self.llm_gateway.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            contains_pii=contains_pii,
        )

        logger.info(f"Generation complete (tokens: {response.total_tokens})")

        return {
            "answer": response.text,
            "model": response.model,
            "backend": response.backend,
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            }
        }

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        return_sources: bool = True,
        contains_pii: bool = False,
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query:
        1. Retrieve relevant documents
        2. Rerank for relevance
        3. Generate answer with LLM
        """
        start_time = datetime.utcnow()

        # Stage 1: Dense retrieval
        documents = await self.retrieve_dense(query, filters=filters)

        if not documents:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
                "latency_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }

        # Stage 2: Reranking
        documents = await self.rerank(query, documents)

        # Build context from top documents
        context = self._build_context(documents)

        # Stage 3: Generate answer
        generation_result = await self.generate(query, context, contains_pii=contains_pii)

        # Build response
        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        response = {
            "answer": generation_result["answer"],
            "model": generation_result["model"],
            "backend": generation_result["backend"],
            "usage": generation_result["usage"],
            "latency_ms": latency_ms,
            "num_documents_retrieved": len(documents),
        }

        if return_sources:
            response["sources"] = [
                {
                    "text": doc.text[:200] + "...",  # Truncate for brevity
                    "metadata": doc.metadata,
                    "relevance_score": doc.score
                }
                for doc in documents
            ]

        logger.info(f"RAG query complete (latency: {latency_ms:.0f}ms)")
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Document Ingestion Utilities
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    document_text: str,
    doc_id: str,
    metadata: Dict[str, Any],
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[Document]:
    """
    Chunk document into overlapping segments (same strategy as original).

    Args:
        document_text: Raw document text
        doc_id: Unique document identifier
        metadata: Document metadata (source, timestamp, etc.)
        chunk_size: Tokens per chunk (~512 chars = ~128 tokens)
        overlap: Overlap between chunks

    Why 512 chars with 50-char overlap?
    - Fits comfortably in LLM context (Gemini 1M tokens, vLLM 8K)
    - 50-char overlap ensures key info not split across chunks
    """
    chunks = []
    # Use character-based chunking (simple, no tokenizer dependency)
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(document_text), step)):
        chunk_text = document_text[start:start + chunk_size]
        if not chunk_text.strip():
            continue
        chunks.append(Document(
            id=f"{doc_id}_chunk_{i}",
            text=chunk_text,
            metadata={
                **metadata,
                "chunk_index": i,
                "doc_id": doc_id,
            }
        ))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (Colab / local)
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """
    Example RAG pipeline usage.
    Run locally or in Google Colab:

    In Colab:
        !pip install qdrant-client sentence-transformers google-generativeai
        import os; os.environ["GOOGLE_API_KEY"] = "YOUR_KEY"
    """
    import os

    config = RAGConfig.from_env()
    # Override for local testing without a running Qdrant
    # Use in-memory Qdrant: QdrantClient(":memory:") in the pipeline
    config.vector_db_host = "localhost"

    pipeline = RAGPipeline(config)

    # Example: Index some documents
    docs = chunk_document(
        document_text=(
            "Machine learning is a subset of artificial intelligence that enables "
            "computers to learn from data without being explicitly programmed. "
            "Neural networks are inspired by biological neurons in the brain."
        ),
        doc_id="ml-intro",
        metadata={"source": "ML Textbook", "chapter": 1}
    )

    await pipeline.add_documents(docs)

    # Example: Query
    result = await pipeline.query("What is machine learning?")
    print(f"Answer: {result['answer']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    print(f"Sources: {len(result.get('sources', []))}")


if __name__ == "__main__":
    asyncio.run(main())
