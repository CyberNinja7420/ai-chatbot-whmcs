"""
RAG (Retrieval-Augmented Generation) Service
Integrates with Qdrant vector database for semantic search.
Indexes WHMCS knowledge base and support ticket history.
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    UpdateStatus
)

from config import get_settings
from services.model_provider import get_model_provider

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document for indexing"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_type: str  # 'knowledge_base', 'ticket', 'faq', etc.
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class SearchResult:
    """A search result from the vector database"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str


@dataclass
class RAGContext:
    """Context assembled for RAG generation"""
    query: str
    relevant_docs: List[SearchResult]
    total_context_length: int
    sources_used: List[str]


class RAGService:
    """
    RAG Service for semantic search and context augmentation.
    Uses Qdrant for vector storage and Ollama for embeddings.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._sync_client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
        self.model_provider = get_model_provider()
        self._collections_initialized = False
    
    @property
    def sync_client(self) -> QdrantClient:
        """Get synchronous Qdrant client"""
        if self._sync_client is None:
            self._sync_client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port
            )
        return self._sync_client
    
    @property
    def async_client(self) -> AsyncQdrantClient:
        """Get asynchronous Qdrant client"""
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port
            )
        return self._async_client
    
    async def initialize_collections(self) -> None:
        """Initialize Qdrant collections if they don't exist"""
        if self._collections_initialized:
            return
        
        collections = [
            self.settings.qdrant_collection_kb,
            self.settings.qdrant_collection_tickets
        ]
        
        try:
            existing = await self.async_client.get_collections()
            existing_names = [c.name for c in existing.collections]
            
            for collection in collections:
                if collection not in existing_names:
                    await self.async_client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(
                            size=self.settings.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection: {collection}")
            
            self._collections_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    def _generate_doc_id(self, content: str, source_type: str, source_id: Optional[str] = None) -> str:
        """Generate a unique document ID"""
        if source_id:
            return f"{source_type}_{source_id}"
        # Hash content for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{source_type}_{content_hash}"
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near chunk boundary
                for boundary in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_boundary = text.rfind(boundary, start + chunk_size // 2, end + 100)
                    if last_boundary != -1:
                        end = last_boundary + len(boundary)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    async def index_document(
        self,
        content: str,
        source_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        chunk_content: bool = True
    ) -> List[str]:
        """
        Index a document into the vector database.
        Returns list of indexed document IDs.
        """
        await self.initialize_collections()
        
        metadata = metadata or {}
        metadata["source_type"] = source_type
        metadata["indexed_at"] = datetime.utcnow().isoformat()
        
        # Determine collection
        collection = (
            self.settings.qdrant_collection_tickets 
            if source_type == "ticket" 
            else self.settings.qdrant_collection_kb
        )
        
        # Chunk if needed
        if chunk_content:
            chunks = self._chunk_text(content)
        else:
            chunks = [content]
        
        indexed_ids = []
        points = []
        
        for i, chunk in enumerate(chunks):
            doc_id = self._generate_doc_id(chunk, source_type, f"{source_id}_{i}" if source_id else None)
            
            # Generate embedding
            try:
                embedding = await self.model_provider.get_embedding(chunk)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                continue
            
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["content_preview"] = chunk[:200]
            
            points.append(PointStruct(
                id=hash(doc_id) % (2**63),  # Qdrant needs integer IDs
                vector=embedding,
                payload={
                    "doc_id": doc_id,
                    "content": chunk,
                    **chunk_metadata
                }
            ))
            indexed_ids.append(doc_id)
        
        if points:
            await self.async_client.upsert(
                collection_name=collection,
                points=points
            )
            logger.info(f"Indexed {len(points)} chunks for {source_type}")
        
        return indexed_ids
    
    async def index_knowledge_article(
        self,
        article_id: str,
        title: str,
        content: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """Index a WHMCS knowledge base article"""
        full_content = f"# {title}\n\n{content}"
        
        metadata = {
            "article_id": article_id,
            "title": title,
            "category": category,
            "tags": tags or [],
            "document_type": "knowledge_article"
        }
        
        return await self.index_document(
            content=full_content,
            source_type="knowledge_base",
            metadata=metadata,
            source_id=f"kb_{article_id}"
        )
    
    async def index_ticket(
        self,
        ticket_id: str,
        subject: str,
        messages: List[Dict[str, str]],
        department: Optional[str] = None,
        status: Optional[str] = None,
        client_id: Optional[str] = None
    ) -> List[str]:
        """Index a support ticket and its conversation"""
        # Combine subject and messages
        content_parts = [f"Subject: {subject}"]
        
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("content", msg.get("message", ""))
            content_parts.append(f"\n[{role}]: {text}")
        
        full_content = "\n".join(content_parts)
        
        metadata = {
            "ticket_id": ticket_id,
            "subject": subject,
            "department": department,
            "status": status,
            "client_id": client_id,
            "message_count": len(messages),
            "document_type": "support_ticket"
        }
        
        return await self.index_document(
            content=full_content,
            source_type="ticket",
            metadata=metadata,
            source_id=f"ticket_{ticket_id}"
        )
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        source_types: Optional[List[str]] = None,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents using semantic similarity.
        """
        await self.initialize_collections()
        
        # Generate query embedding
        try:
            query_embedding = await self.model_provider.get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Build filter conditions
        filter_conditions = []
        
        if source_types:
            filter_conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchValue(value=source_types[0])  # Simplified for single source
                )
            )
        
        if filters:
            for key, value in filters.items():
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        results = []
        
        # Search both collections
        collections = []
        if not source_types or "knowledge_base" in source_types:
            collections.append(self.settings.qdrant_collection_kb)
        if not source_types or "ticket" in source_types:
            collections.append(self.settings.qdrant_collection_tickets)
        
        for collection in collections:
            try:
                search_results = await self.async_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=search_filter,
                    score_threshold=min_score
                )
                
                for hit in search_results:
                    results.append(SearchResult(
                        document_id=hit.payload.get("doc_id", str(hit.id)),
                        content=hit.payload.get("content", ""),
                        score=hit.score,
                        metadata={
                            k: v for k, v in hit.payload.items() 
                            if k not in ["content", "doc_id"]
                        },
                        source_type=hit.payload.get("source_type", "unknown")
                    ))
                    
            except Exception as e:
                logger.warning(f"Search failed for collection {collection}: {e}")
                continue
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def build_context(
        self,
        query: str,
        max_context_length: int = 4000,
        include_knowledge_base: bool = True,
        include_tickets: bool = True,
        client_id: Optional[str] = None
    ) -> RAGContext:
        """
        Build context for RAG generation by searching relevant documents.
        """
        source_types = []
        if include_knowledge_base:
            source_types.append("knowledge_base")
        if include_tickets:
            source_types.append("ticket")
        
        filters = {}
        if client_id:
            filters["client_id"] = client_id
        
        # Search for relevant documents
        results = await self.search(
            query=query,
            limit=10,
            source_types=source_types if source_types else None,
            filters=filters if filters else None
        )
        
        # Build context within length limit
        context_docs = []
        total_length = 0
        sources_used = set()
        
        for result in results:
            doc_length = len(result.content)
            if total_length + doc_length > max_context_length:
                # Try to fit a truncated version
                remaining = max_context_length - total_length
                if remaining > 200:
                    result.content = result.content[:remaining] + "..."
                    context_docs.append(result)
                    total_length += remaining
                    sources_used.add(result.source_type)
                break
            
            context_docs.append(result)
            total_length += doc_length
            sources_used.add(result.source_type)
        
        return RAGContext(
            query=query,
            relevant_docs=context_docs,
            total_context_length=total_length,
            sources_used=list(sources_used)
        )
    
    async def generate_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        client_id: Optional[str] = None,
        model: Optional[str] = None,
        include_knowledge_base: bool = True,
        include_tickets: bool = True
    ) -> Tuple[str, RAGContext]:
        """
        Generate a response using RAG with relevant context.
        Returns the response and the context used.
        """
        # Build context
        context = await self.build_context(
            query=query,
            include_knowledge_base=include_knowledge_base,
            include_tickets=include_tickets,
            client_id=client_id
        )
        
        # Format context for the prompt
        context_text = ""
        if context.relevant_docs:
            context_text = "\n\n---\nRelevant Information:\n"
            for i, doc in enumerate(context.relevant_docs, 1):
                source_info = f"[Source: {doc.source_type}"
                if doc.metadata.get("title"):
                    source_info += f" - {doc.metadata['title']}"
                source_info += f", Relevance: {doc.score:.2f}]"
                
                context_text += f"\n{i}. {source_info}\n{doc.content}\n"
        
        # Build system prompt
        system_prompt = """You are an AI support assistant for a web hosting company using WHMCS.
Your role is to help customers with their hosting-related questions and issues.

Guidelines:
- Be helpful, professional, and concise
- Use the provided context to answer questions accurately
- If the context contains relevant information, cite it in your response
- If you don't have enough information, acknowledge this and suggest next steps
- Never make up information about services, pricing, or technical details
- For billing or account-specific queries, verify information carefully

{context}
""".format(context=context_text)
        
        # Prepare messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        
        # Generate response
        response = await self.model_provider.generate(
            messages=messages,
            model=model,
            system_prompt=system_prompt
        )
        
        return response.content, context
    
    async def delete_document(self, doc_id: str, collection: Optional[str] = None) -> bool:
        """Delete a document from the vector database"""
        collections = [collection] if collection else [
            self.settings.qdrant_collection_kb,
            self.settings.qdrant_collection_tickets
        ]
        
        deleted = False
        for coll in collections:
            try:
                # Find points with matching doc_id
                results = await self.async_client.scroll(
                    collection_name=coll,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    ),
                    limit=100
                )
                
                point_ids = [p.id for p in results[0]]
                if point_ids:
                    await self.async_client.delete(
                        collection_name=coll,
                        points_selector=qdrant_models.PointIdsList(points=point_ids)
                    )
                    deleted = True
                    logger.info(f"Deleted {len(point_ids)} points for doc_id {doc_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to delete from {coll}: {e}")
        
        return deleted
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        await self.initialize_collections()
        
        stats = {}
        for collection in [self.settings.qdrant_collection_kb, self.settings.qdrant_collection_tickets]:
            try:
                info = await self.async_client.get_collection(collection)
                stats[collection] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status.value
                }
            except Exception as e:
                stats[collection] = {"error": str(e)}
        
        return stats


# Singleton instance
_rag_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGService()
    return _rag_instance
