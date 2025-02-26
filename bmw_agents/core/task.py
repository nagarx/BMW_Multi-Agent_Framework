"""
Task and TaskQueue classes for the BMW Agents framework.
These components handle task representation and dependency management.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar

from bmw_agents.utils.logger import get_logger

logger = get_logger("task")

T = TypeVar("T")  # Generic type for task results


class TaskStatus(Enum):
    """Enum representing the possible states of a task."""

    PENDING = "pending"  # Not yet ready to execute (dependencies not met)
    READY = "ready"  # Ready to execute (all dependencies met)
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed to complete
    CANCELLED = "cancelled"  # Cancelled before completion


@dataclass
class Task:
    """
    Represents a task in the agent workflow.

    A task is a unit of work that can have dependencies on other tasks.
    It corresponds to a step in the execution plan created by the Planner agent.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependency_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def is_ready(self) -> bool:
        """
        Check if the task is ready to execute.

        A task is ready when all its dependencies have been completed.

        Returns:
            True if the task is ready, False otherwise
        """
        return self.status == TaskStatus.PENDING and len(self.dependencies) == len(
            self.dependency_results
        )

    def mark_ready(self) -> None:
        """Mark the task as ready for execution."""
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.READY
            logger.debug(f"Task {self.id} is now ready for execution")

    def mark_running(self) -> None:
        """Mark the task as currently running."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        logger.debug(f"Task {self.id} is now running")

    def mark_completed(self, result: Any) -> None:
        """
        Mark the task as completed with the given result.

        Args:
            result: The result of the task execution
        """
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
        logger.debug(f"Task {self.id} completed successfully")

    def mark_failed(self, error: str) -> None:
        """
        Mark the task as failed with the given error.

        Args:
            error: The error message
        """
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()
        logger.error(f"Task {self.id} failed: {error}")

    def mark_cancelled(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
        logger.warning(f"Task {self.id} was cancelled")

    def add_dependency_result(self, task_id: str, result: Any) -> None:
        """
        Add the result of a dependency task.

        Args:
            task_id: The ID of the dependency task
            result: The result of the dependency task
        """
        if task_id in self.dependencies:
            self.dependency_results[task_id] = result
            logger.debug(f"Added result from dependency {task_id} to task {self.id}")
        else:
            logger.warning(
                f"Attempted to add result for non-dependency {task_id} to task {self.id}"
            )

    def can_retry(self) -> bool:
        """
        Check if the task can be retried.

        Returns:
            True if the task can be retried, False otherwise
        """
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def retry(self) -> None:
        """Reset the task for retry."""
        if self.can_retry():
            self.status = TaskStatus.READY
            self.retry_count += 1
            self.error = None
            self.started_at = None
            logger.info(f"Retrying task {self.id} (attempt {self.retry_count}/{self.max_retries})")

    def get_duration(self) -> Optional[float]:
        """
        Get the duration of the task execution in seconds.

        Returns:
            The duration if the task has completed, None otherwise
        """
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary representation.

        Returns:
            Dictionary representation of the task
        """
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Create a task from a dictionary representation.

        Args:
            data: Dictionary representation of the task

        Returns:
            The created task
        """
        # Convert status string to enum
        status_str = data.pop("status", "pending")
        status = TaskStatus(status_str)

        return cls(status=status, **data)


class TaskQueue:
    """
    Queue for managing tasks and their dependencies.

    The task queue is responsible for tracking tasks, managing their dependencies,
    and determining which tasks are ready for execution.
    """

    def __init__(self) -> None:
        """Initialize an empty task queue."""
        self.tasks: Dict[str, Task] = {}
        self.ready_tasks: List[str] = []

        # Track which tasks have been executed
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()

    def add_task(self, task: Task) -> None:
        """
        Add a task to the queue.

        Args:
            task: The task to add
        """
        self.tasks[task.id] = task

        # If the task has no dependencies, it's ready to execute
        if not task.dependencies:
            task.mark_ready()
            self.ready_tasks.append(task.id)

        logger.debug(f"Added task {task.id} to queue")

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.

        Args:
            task_id: The ID of the task to get

        Returns:
            The task if found, None otherwise
        """
        return self.tasks.get(task_id)

    def get_ready_tasks(self) -> List[Task]:
        """
        Get all tasks that are ready to execute.

        Returns:
            List of ready tasks
        """
        return [self.tasks[task_id] for task_id in self.ready_tasks]

    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task that is ready to execute.

        Returns:
            The next ready task, or None if no tasks are ready
        """
        if not self.ready_tasks:
            return None

        task_id = self.ready_tasks[0]
        return self.tasks[task_id]

    def update_task_status(
        self, task_id: str, status: TaskStatus, result: Any = None, error: str = None
    ) -> None:
        """
        Update the status of a task and propagate results to dependent tasks.

        Args:
            task_id: The ID of the task to update
            status: The new status
            result: The result of the task execution (if completed)
            error: The error message (if failed)
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Attempted to update non-existent task: {task_id}")
            return

        # Update the task status
        if status == TaskStatus.COMPLETED:
            task.mark_completed(result)
            self.completed_tasks.add(task_id)

            # Remove from ready list if it's there
            if task_id in self.ready_tasks:
                self.ready_tasks.remove(task_id)

            # Propagate result to dependent tasks
            self._propagate_result(task_id, result)

        elif status == TaskStatus.FAILED:
            task.mark_failed(error)
            self.failed_tasks.add(task_id)

            # Remove from ready list if it's there
            if task_id in self.ready_tasks:
                self.ready_tasks.remove(task_id)

        elif status == TaskStatus.RUNNING:
            task.mark_running()

            # Remove from ready list if it's there
            if task_id in self.ready_tasks:
                self.ready_tasks.remove(task_id)

        elif status == TaskStatus.READY:
            task.mark_ready()

            # Add to ready list if not already there
            if task_id not in self.ready_tasks:
                self.ready_tasks.append(task_id)

        elif status == TaskStatus.CANCELLED:
            task.mark_cancelled()

            # Remove from ready list if it's there
            if task_id in self.ready_tasks:
                self.ready_tasks.remove(task_id)

    def retry_task(self, task_id: str) -> bool:
        """
        Retry a failed task.

        Args:
            task_id: The ID of the task to retry

        Returns:
            True if the task was retried, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Attempted to retry non-existent task: {task_id}")
            return False

        if task.can_retry():
            task.retry()
            self.ready_tasks.append(task_id)
            self.failed_tasks.remove(task_id)
            return True
        else:
            logger.warning(f"Task {task_id} cannot be retried")
            return False

    def _propagate_result(self, task_id: str, result: Any) -> None:
        """
        Propagate the result of a completed task to its dependents.

        Args:
            task_id: The ID of the completed task
            result: The result of the completed task
        """
        for dependent_task in self.tasks.values():
            if task_id in dependent_task.dependencies:
                dependent_task.add_dependency_result(task_id, result)

                # Check if the dependent task is now ready
                if dependent_task.is_ready():
                    dependent_task.mark_ready()
                    self.ready_tasks.append(dependent_task.id)

    def is_complete(self) -> bool:
        """
        Check if all tasks have been completed.

        Returns:
            True if all tasks are completed, False otherwise
        """
        return (
            len(self.completed_tasks) + len(self.failed_tasks) == len(self.tasks)
            and len(self.tasks) > 0
        )

    def all_completed_successfully(self) -> bool:
        """
        Check if all tasks have been completed successfully.

        Returns:
            True if all tasks completed successfully, False otherwise
        """
        return len(self.completed_tasks) == len(self.tasks) and len(self.tasks) > 0

    def get_completion_percentage(self) -> float:
        """
        Get the percentage of tasks that have been completed.

        Returns:
            Percentage of completed tasks (0-100)
        """
        if not self.tasks:
            return 0.0

        completed = len(self.completed_tasks) + len(self.failed_tasks)
        return (completed / len(self.tasks)) * 100

    def reset(self) -> None:
        """Reset the task queue to its initial state."""
        self.tasks = {}
        self.ready_tasks = []
        self.completed_tasks = set()
        self.failed_tasks = set()

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Get all tasks with the specified status.

        Args:
            status: The status to filter by

        Returns:
            List of tasks with the specified status
        """
        return [task for task in self.tasks.values() if task.status == status]
