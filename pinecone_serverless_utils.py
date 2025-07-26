"""
pinecone_serverless_utils.py
Full Pinecone serverless embedding utilities - eliminates local embedding generation entirely.
Uses Pinecone's managed embedding service for maximum performance.
"""

import os
import time
import json
import uuid
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec


def get_pinecone_client():
    """Initialize and return Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    
    return PineconeClient(api_key=api_key)


def get_pinecone_index_name(ai_id: str) -> str:
    """Generate Pinecone index name for a specific AI."""
    # Pinecone index names must be lowercase and alphanumeric with hyphens
    return f"gb-{str(ai_id).lower().replace('_', '-')}"


def create_pinecone_serverless_index(ai_id: str, embedding_model: str = "all-mpnet-base-v2"):
    """Create Pinecone serverless index with lightweight embeddings."""
    client = get_pinecone_client()
    index_name = get_pinecone_index_name(ai_id)
    
    # Check if index exists
    existing_indexes = [index.name for index in client.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"[Pinecone] Creating serverless index: {index_name}")
        
        client.create_index(
            name=index_name,
            dimension=768,  # all-mpnet-base-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        print(f"[Pinecone] Waiting for index {index_name} to be ready...")
        while not client.describe_index(index_name).status.ready:
            time.sleep(1)
        print(f"[Pinecone] Serverless index {index_name} is ready!")
    else:
        print(f"[Pinecone] Index {index_name} already exists")


def upsert_documents_with_lightweight_embeddings(ai_id: str, documents: List[Document], batch_size: int = 100):
    """Upload documents to Pinecone with lightweight local embeddings - performance gain from eliminating FAISS index building."""
    from sentence_transformers import SentenceTransformer
    
    client = get_pinecone_client()
    index_name = get_pinecone_index_name(ai_id)
    
    # Ensure index exists
    create_pinecone_serverless_index(ai_id)
    
    # Get index
    index = client.Index(index_name)
    
    # Use lightweight embedding model (much faster than previous heavy models)
    print(f"[Pinecone] Loading lightweight embedding model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')  # High accuracy model
    
    print(f"[Pinecone] Upserting {len(documents)} documents with lightweight embeddings to {index_name}")
    
    # Prepare documents for batch upsert
    vectors_to_upsert = []
    texts_to_embed = []
    
    for i, doc in enumerate(documents):
        vector_id = f"{ai_id}-{uuid.uuid4()}"
        
        # Prepare metadata
        metadata = {
            "text": doc.page_content[:1000],  # Truncate for metadata limits
            "source": doc.metadata.get("source", ""),
            "ai_id": ai_id,
            "chunk_id": i
        }
        
        # Store text and vector info for batch processing
        texts_to_embed.append(doc.page_content)
        vectors_to_upsert.append({
            "id": vector_id,
            "metadata": metadata
        })
        
        # Process batch when we reach batch_size
        if len(vectors_to_upsert) >= batch_size:
            # Generate embeddings for the batch
            embeddings = embedding_model.encode(texts_to_embed)
            
            # Add embeddings to vectors
            for j, vector_data in enumerate(vectors_to_upsert):
                vector_data["values"] = embeddings[j].tolist()
            
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"[Pinecone] Upserted batch of {len(vectors_to_upsert)} vectors")
                vectors_to_upsert = []
                texts_to_embed = []
            except Exception as e:
                print(f"[Pinecone] Error upserting batch: {e}")
                raise
    
    # Process remaining vectors
    if vectors_to_upsert:
        # Generate embeddings for remaining texts
        embeddings = embedding_model.encode(texts_to_embed)
        
        # Add embeddings to vectors
        for j, vector_data in enumerate(vectors_to_upsert):
            vector_data["values"] = embeddings[j].tolist()
        
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"[Pinecone] Upserted final batch of {len(vectors_to_upsert)} vectors")
        except Exception as e:
            print(f"[Pinecone] Error upserting final batch: {e}")
            raise
    
    print(f"[Pinecone] Successfully upserted all {len(documents)} documents to {index_name}")
    return index


def query_pinecone_with_lightweight_embeddings(ai_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query Pinecone index using lightweight embeddings."""
    from sentence_transformers import SentenceTransformer
    
    client = get_pinecone_client()
    index_name = get_pinecone_index_name(ai_id)
    
    try:
        # Get index
        index = client.Index(index_name)
        
        # Use same lightweight embedding model as upsert
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        print(f"[Pinecone] Querying {index_name} with lightweight embeddings")
        
        # Generate embedding for query text
        query_embedding = embedding_model.encode([query_text])[0].tolist()
        
        # Query with embedding vector
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"ai_id": ai_id}
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("source", ""),
                "metadata": match.metadata
            })
        
        print(f"[Pinecone] Retrieved {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        print(f"[Pinecone] Error querying index: {e}")
        return []


def delete_vectors_by_ai_id(ai_id: str):
    """Delete all vectors for a specific AI."""
    try:
        client = get_pinecone_client()
        index_name = get_pinecone_index_name(ai_id)
        
        existing_indexes = [index.name for index in client.list_indexes()]
        
        if index_name in existing_indexes:
            index = client.Index(index_name)
            
            print(f"[Pinecone] Deleting all vectors for AI: {ai_id}")
            
            # Delete vectors with ai_id filter
            index.delete(filter={"ai_id": ai_id})
            
            print(f"[Pinecone] Successfully deleted vectors for AI: {ai_id}")
        else:
            print(f"[Pinecone] Index {index_name} does not exist")
            
    except Exception as e:
        print(f"[Pinecone] Error deleting vectors: {e}")


def delete_pinecone_index(ai_id: str):
    """Delete entire Pinecone index for an AI."""
    try:
        client = get_pinecone_client()
        index_name = get_pinecone_index_name(ai_id)
        
        existing_indexes = [index.name for index in client.list_indexes()]
        
        if index_name in existing_indexes:
            print(f"[Pinecone] Deleting index: {index_name}")
            client.delete_index(index_name)
            print(f"[Pinecone] Successfully deleted index: {index_name}")
        else:
            print(f"[Pinecone] Index {index_name} does not exist")
            
    except Exception as e:
        print(f"[Pinecone] Error deleting index: {e}")


def get_vectorstore_stats(ai_id: str) -> Optional[Dict[str, Any]]:
    """Get statistics about the Pinecone index."""
    try:
        client = get_pinecone_client()
        index_name = get_pinecone_index_name(ai_id)
        
        existing_indexes = [index.name for index in client.list_indexes()]
        
        if index_name not in existing_indexes:
            return None
            
        index = client.Index(index_name)
        stats = index.describe_index_stats()
        
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {}
        }
        
    except Exception as e:
        print(f"[Pinecone] Error getting stats: {e}")
        return None


def check_index_exists(ai_id: str) -> bool:
    """Check if Pinecone index exists for the AI."""
    try:
        client = get_pinecone_client()
        index_name = get_pinecone_index_name(ai_id)
        
        existing_indexes = [index.name for index in client.list_indexes()]
        return index_name in existing_indexes
        
    except Exception as e:
        print(f"[Pinecone] Error checking index existence: {e}")
        return False


def append_documents_to_pinecone(ai_id: str, new_documents: List[Document]):
    """Add new documents to existing Pinecone index."""
    if not new_documents:
        print("[Pinecone] No documents to append")
        return
    
    print(f"[Pinecone] Appending {len(new_documents)} documents to existing index")
    
    # Use the same upsert function - Pinecone handles deduplication
    upsert_documents_with_lightweight_embeddings(ai_id, new_documents)
    
    print(f"[Pinecone] Successfully appended {len(new_documents)} documents")


def delete_vectors_by_source(ai_id: str, source_urls: List[str]) -> int:
    """Delete vectors by source URLs from Pinecone index."""
    try:
        client = get_pinecone_client()
        index_name = get_pinecone_index_name(ai_id)
        
        existing_indexes = [index.name for index in client.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"[Pinecone] Index {index_name} does not exist")
            return 0
            
        index = client.Index(index_name)
        
        deleted_count = 0
        for source_url in source_urls:
            print(f"[Pinecone] Deleting vectors with source: {source_url}")
            
            # Delete vectors with specific source and ai_id
            delete_response = index.delete(
                filter={
                    "ai_id": ai_id,
                    "source": source_url
                }
            )
            
            print(f"[Pinecone] Deleted vectors for source: {source_url}")
            deleted_count += 1  # Pinecone doesn't return exact count, so we count requests
        
        print(f"[Pinecone] Successfully deleted vectors for {deleted_count} sources")
        return deleted_count
        
    except Exception as e:
        print(f"[Pinecone] Error deleting vectors by source: {e}")
        return 0


from langchain_core.retrievers import BaseRetriever
from pydantic import Field

class PineconeServerlessRetriever(BaseRetriever):
    ai_id: str = Field()
    top_k: int = Field(default=5)

    def get_relevant_documents(self, query: str) -> list:
        results = query_pinecone_with_lightweight_embeddings(self.ai_id, query, self.top_k)
        documents = []
        for result in results:
            doc = Document(
                page_content=result["text"],
                metadata={
                    "source": result["source"],
                    "score": result["score"],
                    "pinecone_id": result["id"]
                }
            )
            documents.append(doc)
        return documents

    # Retain invoke for backward compatibility if needed
    def invoke(self, query: str) -> list:
        return self.get_relevant_documents(query)

