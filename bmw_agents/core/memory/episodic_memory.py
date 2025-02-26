"""
Episodic memory module for the BMW Agents framework.
This module implements the episodic memory used to store and retrieve cross-task knowledge.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

from bmw_agents.utils.logger import get_logger

logger = get_logger("memory.episodic_memory")


@dataclass
class Episode:
    """
    Represents a single episode in episodic memory.

    An episode corresponds to a completed task, with all associated metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    workflow_id: str = ""
    task_description: str = ""
    task_result: str = ""
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create an episode from a dictionary."""
        return cls(**data)


class EpisodicMemory:
    """
    Episodic memory for storing and retrieving task episodes.

    Episodic Memory (EM) is a container that keeps records of completed tasks
    across applications. It uses a vector database for semantic retrieval.
    """

    def __init__(
        self,
        persist_directory: str = ".bmw_agents_memory",
        embedding_function: Optional[EmbeddingFunction] = None,
    ) -> None:
        """
        Initialize episodic memory.

        Args:
            persist_directory: Directory to persist the vector database
            embedding_function: Function to use for embedding (default: OpenAI's)
        """
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Set up embedding function
        if embedding_function is None:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY", ""), model_name="text-embedding-ada-002"
            )
        else:
            self.embedding_function = embedding_function

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collections
        self.description_collection = self.client.get_or_create_collection(
            name="task_descriptions",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self.result_collection = self.client.get_or_create_collection(
            name="task_results",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory cache for episodes
        self.episodes_cache: Dict[str, Episode] = {}

        logger.info(f"Initialized episodic memory with {self.count()} episodes")

    def add(self, episode: Episode) -> None:
        """
        Add an episode to the memory.

        Args:
            episode: The episode to add
        """
        # Add to description collection
        self.description_collection.add(
            ids=[episode.id],
            documents=[episode.task_description],
            metadatas=[
                {
                    "task_id": episode.task_id,
                    "workflow_id": episode.workflow_id,
                    "created_at": episode.created_at,
                }
            ],
        )

        # Add to result collection
        self.result_collection.add(
            ids=[episode.id],
            documents=[episode.task_result],
            metadatas=[
                {
                    "task_id": episode.task_id,
                    "workflow_id": episode.workflow_id,
                    "created_at": episode.created_at,
                }
            ],
        )

        # Update cache
        self.episodes_cache[episode.id] = episode

        logger.debug(f"Added episode {episode.id} to memory")

    def get(self, episode_id: str) -> Optional[Episode]:
        """
        Get an episode by its ID.

        Args:
            episode_id: The ID of the episode to get

        Returns:
            The episode if found, None otherwise
        """
        # Check cache first
        if episode_id in self.episodes_cache:
            return self.episodes_cache[episode_id]

        # Query collections
        description_result = self.description_collection.get(ids=[episode_id])
        result_result = self.result_collection.get(ids=[episode_id])

        if not description_result["documents"] or not result_result["documents"]:
            return None

        # Create Episode object
        metadata = description_result["metadatas"][0]
        episode = Episode(
            id=episode_id,
            task_id=metadata.get("task_id", ""),
            workflow_id=metadata.get("workflow_id", ""),
            task_description=description_result["documents"][0],
            task_result=result_result["documents"][0],
            created_at=metadata.get("created_at", time.time()),
        )

        # Update cache
        self.episodes_cache[episode_id] = episode

        return episode

    def query_by_description(
        self,
        query: str,
        n_results: int = 5,
        workflow_id: Optional[str] = None,
        min_score: float = 0.6,
    ) -> List[Episode]:
        """
        Query episodes by description similarity.

        Args:
            query: The query string
            n_results: Number of results to return
            workflow_id: Filter by workflow ID (optional)
            min_score: Minimum similarity score

        Returns:
            List of matching episodes
        """
        # Prepare where clause if workflow_id is provided
        where_clause = {"workflow_id": workflow_id} if workflow_id else None

        # Query description collection
        results = self.description_collection.query(
            query_texts=[query], n_results=n_results, where=where_clause
        )

        episodes = []
        for i, episode_id in enumerate(results["ids"][0]):
            # Get distance and convert to similarity score (1 - distance for cosine)
            score = 1 - results["distances"][0][i] if "distances" in results else 0

            # Skip if score is below threshold
            if score < min_score:
                continue

            episode = self.get(episode_id)
            if episode:
                episodes.append(episode)

        return episodes

    def query_by_result(
        self,
        query: str,
        n_results: int = 5,
        workflow_id: Optional[str] = None,
        min_score: float = 0.6,
    ) -> List[Episode]:
        """
        Query episodes by result similarity.

        Args:
            query: The query string
            n_results: Number of results to return
            workflow_id: Filter by workflow ID (optional)
            min_score: Minimum similarity score

        Returns:
            List of matching episodes
        """
        # Prepare where clause if workflow_id is provided
        where_clause = {"workflow_id": workflow_id} if workflow_id else None

        # Query result collection
        results = self.result_collection.query(
            query_texts=[query], n_results=n_results, where=where_clause
        )

        episodes = []
        for i, episode_id in enumerate(results["ids"][0]):
            # Get distance and convert to similarity score (1 - distance for cosine)
            score = 1 - results["distances"][0][i] if "distances" in results else 0

            # Skip if score is below threshold
            if score < min_score:
                continue

            episode = self.get(episode_id)
            if episode:
                episodes.append(episode)

        return episodes

    def get_by_task_id(self, task_id: str) -> Optional[Episode]:
        """
        Get an episode by its task ID.

        Args:
            task_id: The task ID to search for

        Returns:
            The episode if found, None otherwise
        """
        # Query description collection
        results = self.description_collection.get(where={"task_id": task_id})

        if not results["ids"]:
            return None

        return self.get(results["ids"][0])

    def get_by_workflow_id(self, workflow_id: str) -> List[Episode]:
        """
        Get all episodes for a workflow.

        Args:
            workflow_id: The workflow ID to search for

        Returns:
            List of episodes for the workflow
        """
        # Query description collection
        results = self.description_collection.get(where={"workflow_id": workflow_id})

        episodes = []
        for episode_id in results["ids"]:
            episode = self.get(episode_id)
            if episode:
                episodes.append(episode)

        return episodes

    def get_most_relevant(
        self, query: str, n_results: int = 3, workflow_id: Optional[str] = None
    ) -> List[Episode]:
        """
        Get the most relevant episodes for a query.

        This checks both description and result collections and returns
        the combined most relevant episodes.

        Args:
            query: The query string
            n_results: Number of results to return
            workflow_id: Filter by workflow ID (optional)

        Returns:
            List of most relevant episodes
        """
        # Query both collections
        desc_episodes = self.query_by_description(
            query, n_results=n_results * 2, workflow_id=workflow_id  # Get more and filter later
        )

        result_episodes = self.query_by_result(
            query, n_results=n_results * 2, workflow_id=workflow_id  # Get more and filter later
        )

        # Combine and deduplicate
        seen_ids = set()
        combined_episodes = []

        # Interleave results from both queries
        for i in range(max(len(desc_episodes), len(result_episodes))):
            if i < len(desc_episodes) and desc_episodes[i].id not in seen_ids:
                combined_episodes.append(desc_episodes[i])
                seen_ids.add(desc_episodes[i].id)

            if i < len(result_episodes) and result_episodes[i].id not in seen_ids:
                combined_episodes.append(result_episodes[i])
                seen_ids.add(result_episodes[i].id)

            if len(combined_episodes) >= n_results:
                break

        return combined_episodes[:n_results]

    def delete(self, episode_id: str) -> bool:
        """
        Delete an episode from memory.

        Args:
            episode_id: The ID of the episode to delete

        Returns:
            True if the episode was deleted, False otherwise
        """
        try:
            # Delete from collections
            self.description_collection.delete(ids=[episode_id])
            self.result_collection.delete(ids=[episode_id])

            # Remove from cache
            if episode_id in self.episodes_cache:
                del self.episodes_cache[episode_id]

            logger.debug(f"Deleted episode {episode_id} from memory")
            return True
        except Exception as e:
            logger.error(f"Failed to delete episode {episode_id}: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all episodes from memory.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear collections
            self.description_collection.delete(where={})
            self.result_collection.delete(where={})

            # Clear cache
            self.episodes_cache.clear()

            logger.info("Cleared all episodes from memory")
            return True
        except Exception as e:
            logger.error(f"Failed to clear episodic memory: {e}")
            return False

    def count(self) -> int:
        """
        Get the number of episodes in memory.

        Returns:
            Number of episodes
        """
        return len(self.description_collection.get()["ids"])

    def export(self, path: str) -> bool:
        """
        Export all episodes to a JSON file.

        Args:
            path: Path to the output file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all episodes
            episode_ids = self.description_collection.get()["ids"]
            episodes = [self.get(episode_id).to_dict() for episode_id in episode_ids]

            # Write to file
            with open(path, "w") as f:
                json.dump(episodes, f, indent=2)

            logger.info(f"Exported {len(episodes)} episodes to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export episodes: {e}")
            return False

    def import_from_file(self, path: str) -> int:
        """
        Import episodes from a JSON file.

        Args:
            path: Path to the input file

        Returns:
            Number of imported episodes
        """
        try:
            # Read from file
            with open(path, "r") as f:
                episodes_data = json.load(f)

            # Import episodes
            imported_count = 0
            for episode_data in episodes_data:
                episode = Episode.from_dict(episode_data)
                self.add(episode)
                imported_count += 1

            logger.info(f"Imported {imported_count} episodes from {path}")
            return imported_count
        except Exception as e:
            logger.error(f"Failed to import episodes: {e}")
            return 0
