"""
Architecture View Generation task module.

Based on the CodeToDiagram study: generate PlantUML architecture diagrams from
repository summaries and evaluate them against ground truth diagram images using
image similarity metrics and an LLM-as-a-judge.
"""

from archbench.tasks.diagram import dataset, prompts, grading

__all__ = ["dataset", "prompts", "grading"]
