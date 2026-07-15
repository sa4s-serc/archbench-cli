"""
ArchBench task modules.

Each task has its own module with:
- dataset.py: Data loading
- prompts.py: Prompt templates
- grading.py: Evaluation metrics
"""

from archbench.constants import TASKS, TaskType
from archbench.tasks import adr, traceability, diagram

__all__ = ["TASKS", "TaskType", "adr", "traceability", "diagram"]
