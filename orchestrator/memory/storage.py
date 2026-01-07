"""
Memory Storage using ChromaDB

Provides semantic memory storage with vector embeddings for the voice agent.
"""
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger(__name__)

# Global singleton instance
_memory_storage_instance: Optional["MemoryStorage"] = None


class MemoryStorage:
    """ChromaDB-based memory storage with semantic search."""

    def __init__(self, storage_path: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initialize ChromaDB storage.

        Args:
            storage_path: Path to store ChromaDB data (defaults to config)
            embedding_model: Sentence-transformers model name (defaults to config)
        """
        settings = get_settings()

        # Use provided values or fall back to config
        self.storage_path = Path(storage_path or settings.orchestrator.memory.storage_path)
        self.embedding_model = embedding_model or settings.orchestrator.memory.embedding_model

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing MemoryStorage at {self.storage_path}")
        logger.info(f"Using embedding model: {self.embedding_model}")

        # Initialize ChromaDB client with persistence
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.storage_path),
            anonymized_telemetry=False
        ))

        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="memories",
            embedding_function=self.embedding_function,
            metadata={"description": "Voice agent conversation memories"}
        )

        logger.info(f"MemoryStorage initialized with {self.collection.count()} existing memories")

    async def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        source: str = "conversation"
    ) -> str:
        """
        Add a memory to storage.

        Args:
            content: Text content to store
            tags: Optional list of tags for categorization
            source: Source of the memory (default: "conversation")

        Returns:
            ID of stored memory
        """
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")

        memory_id = str(uuid.uuid4())

        # Build metadata
        metadata = {
            "timestamp": datetime.now().timestamp(),
            "source": source
        }

        # Add tags to metadata (ChromaDB requires string values)
        if tags:
            metadata["tags"] = ",".join(tags)

        # Add to collection
        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata]
        )

        logger.info(f"Added memory {memory_id[:8]}... with tags: {tags}")
        return memory_id

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories semantically using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return (default: 5, max: 20)
            tags: Optional tag filter (returns memories with ANY of these tags)

        Returns:
            List of matching memories with metadata and relevance scores
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Cap limit at 20
        limit = min(limit, 20)

        # Build where clause for tag filtering if needed
        where = None
        if tags:
            # For a single tag, use simple contains
            if len(tags) == 1:
                where = {"tags": {"$contains": tags[0]}}
            else:
                # For multiple tags, use $or to match ANY tag
                # Note: This is a workaround since ChromaDB's metadata filtering is limited
                # We'll filter in Python after retrieving results
                pass

        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=limit if not tags or len(tags) == 1 else limit * 2,  # Get more if filtering
            where=where
        )

        # Format results
        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]

                # Multi-tag filtering (if needed)
                if tags and len(tags) > 1:
                    memory_tags_str = metadata.get("tags", "")
                    if memory_tags_str:
                        memory_tags = set(tag.strip() for tag in memory_tags_str.split(","))
                        # Check if any of the query tags match
                        if not any(tag in memory_tags for tag in tags):
                            continue

                # Parse tags from metadata
                tags_str = metadata.get("tags", "")
                tag_list = [tag.strip() for tag in tags_str.split(",")] if tags_str else []

                memories.append({
                    "id": memory_id,
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else 0.0,
                    "tags": tag_list,
                    "timestamp": metadata.get("timestamp", 0),
                    "source": metadata.get("source", "unknown")
                })

                # Stop if we've hit the limit after filtering
                if len(memories) >= limit:
                    break

        logger.info(f"Found {len(memories)} memories for query: {query[:50]}...")
        return memories

    async def list_all_tags(self) -> Dict[str, int]:
        """
        Get all unique tags with their counts.

        Returns:
            Dictionary mapping tag name to count
        """
        # Get all metadata from the collection
        all_data = self.collection.get()

        tag_counts = {}
        for metadata in all_data["metadatas"]:
            tags_str = metadata.get("tags", "")
            if tags_str:
                for tag in tags_str.split(","):
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return tag_counts

    async def list_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all stored memories for inspection.

        Returns:
            List of all memories with their metadata
        """
        # Get all data from collection
        all_data = self.collection.get()

        memories = []
        if all_data["ids"]:
            for i, memory_id in enumerate(all_data["ids"]):
                metadata = all_data["metadatas"][i]

                # Parse tags
                tags_str = metadata.get("tags", "")
                tag_list = [tag.strip() for tag in tags_str.split(",")] if tags_str else []

                memories.append({
                    "id": memory_id,
                    "content": all_data["documents"][i],
                    "tags": tag_list,
                    "timestamp": metadata.get("timestamp", 0),
                    "source": metadata.get("source", "unknown")
                })

        # Sort by timestamp (newest first)
        memories.sort(key=lambda x: x["timestamp"], reverse=True)

        return memories

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory {memory_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def close(self):
        """Persist and close ChromaDB."""
        try:
            # ChromaDB with duckdb+parquet auto-persists
            # Just log for confirmation
            logger.info(f"MemoryStorage persisted {self.collection.count()} memories")
        except Exception as e:
            logger.error(f"Error during MemoryStorage close: {e}", exc_info=True)


def get_memory_storage() -> MemoryStorage:
    """
    Get the global MemoryStorage singleton instance.

    Returns:
        MemoryStorage instance
    """
    global _memory_storage_instance

    if _memory_storage_instance is None:
        _memory_storage_instance = MemoryStorage()

    return _memory_storage_instance
