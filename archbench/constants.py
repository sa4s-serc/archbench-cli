"""
Constants and configuration for ArchBench.
"""

from enum import Enum
from pathlib import Path
from typing import TypedDict, List, Optional

# =============================================================================
# Task Definitions
# =============================================================================

class TaskType(Enum):
    """Available benchmark tasks."""
    ADR = "adr"                      # Architecture Decision Record Generation
    SERVERLESS = "serverless"        # Serverless Component Generation
    DYNAMIC = "dynamic"              # Dynamic IoT Service Generation
    TRACEABILITY = "traceability"    # Architecture Traceability Link Recovery
    DIAGRAM = "diagram"              # Architecture View Generation


TASKS = {
    TaskType.ADR.value: {
        "name": "Architecture Decision Record Generation",
        "description": "Generate ADR decisions from architectural context",
        "metrics": ["rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore_p", "bertscore_r", "bertscore_f1"],
        "primary_metric": "bertscore_f1",
        "dataset": "sa4s-serc/archbench-adr",
    },
    TaskType.SERVERLESS.value: {
        "name": "Serverless Component Generation",
        "description": "Generate serverless functions from specifications",
        "metrics": ["codebase_tests", "function_tests", "codebleu", "cyclomatic_complexity"],
        "primary_metric": "function_tests",
        "dataset": "sa4s-serc/archbench-serverless",
    },
    TaskType.DYNAMIC.value: {
        "name": "Dynamic IoT Service Generation",
        "description": "Generate IoT services dynamically at runtime",
        "metrics": ["codebertscore_p", "codebertscore_r", "codebertscore_f1"],
        "primary_metric": "codebertscore_f1",
        "dataset": "sa4s-serc/archbench-dynamic",
    },
    TaskType.TRACEABILITY.value: {
        "name": "Architecture Traceability Link Recovery",
        "description": "Recover traceability links between documentation and code",
        "metrics": ["precision", "recall", "f1"],
        "primary_metric": "f1",
        "dataset": "sa4s-serc/archbench-traceability",
    },
    TaskType.DIAGRAM.value: {
        "name": "Architecture View Generation",
        "description": "Generate architecture diagrams (PlantUML) from repository summaries",
        "metrics": ["ssim", "psnr", "rmse", "sam", "sre", "uiq"],
        "primary_metric": "ssim",
        "dataset": "sa4s-serc/archbench-diagram",
    },
}

# =============================================================================
# Data Keys
# =============================================================================

KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "prediction"          # The generated output
KEY_RAW_OUTPUT = "raw_output"           # Raw LLM response before parsing
KEY_REFERENCE = "reference"             # Ground truth

# ADR-specific keys
KEY_CONTEXT = "context"
KEY_DECISION = "decision"

# Traceability-specific keys
KEY_TRACE_LINKS = "trace_links"
KEY_DOCUMENTATION = "documentation"
KEY_CODE_ARTIFACTS = "code_artifacts"

# Diagram-specific keys
KEY_SUMMARY = "summary"                     # Repository summary (inference input)
KEY_CONCERN = "concern"                     # Architectural concern to capture
KEY_BEHAVIOR = "behavior"                   # System behavior (static / dynamic)
KEY_GROUND_TRUTH_IMAGE = "ground_truth_image"   # Path to reference diagram image
KEY_GENERATED_IMAGE = "generated_image"     # Path to generated diagram image

# =============================================================================
# Instance Types
# =============================================================================

class ADRInstance(TypedDict):
    """Schema for ADR task instances."""
    instance_id: str
    context: str              # The architectural context
    decision: str             # Ground truth decision (reference)


class TraceabilityInstance(TypedDict):
    """Schema for traceability task instances."""
    instance_id: str
    project: str
    documentation: str
    code_artifacts: List[str]
    trace_links: List[dict]   # Ground truth links


class DiagramInstance(TypedDict):
    """Schema for architecture view generation task instances."""
    instance_id: str
    summary: str              # Repository summary (inference input)
    concern: str              # Architectural concern to capture
    behavior: str             # System behavior (static / dynamic)
    ground_truth_image: str   # Path to reference diagram image (reference)


class Prediction(TypedDict):
    """Schema for model predictions."""
    instance_id: str
    model_name_or_path: str
    prediction: str
    raw_output: Optional[str]


# =============================================================================
# Logging Constants
# =============================================================================

LOG_DIR = Path("logs")
LOG_INFERENCE_DIR = LOG_DIR / "inference"
LOG_EVALUATION_DIR = LOG_DIR / "evaluation"

LOG_TRAJECTORY = "trajectory.jsonl"     # Step-by-step reasoning
LOG_REPORT = "report.json"              # Final evaluation report
LOG_PREDICTIONS = "predictions.jsonl"   # All predictions
LOG_METRICS = "metrics.json"            # Per-instance metrics

# =============================================================================
# Evaluation Status
# =============================================================================

class EvalStatus(Enum):
    """Evaluation status for an instance."""
    SUCCESS = "success"
    ERROR = "error"
    MISSING_PREDICTION = "missing_prediction"
    INVALID_FORMAT = "invalid_format"


# =============================================================================
# Prompt Templates (for reference - actual prompts in inference/prompts.py)
# =============================================================================

PROMPT_STYLES = {
    "zero_shot": "Direct prompting without examples",
    "few_shot": "Prompting with 2 examples",
    "cot": "Chain-of-thought prompting",
}

# =============================================================================
# Metric Thresholds (for determining "resolved" status)
# =============================================================================

METRIC_THRESHOLDS = {
    TaskType.ADR.value: {
        "bertscore_f1": 0.85,  # Considered good if BERTScore F1 > 0.85
    },
    TaskType.TRACEABILITY.value: {
        "f1": 0.70,  # Considered good if F1 > 0.70
    },
    TaskType.DIAGRAM.value: {
        "ssim": 0.70,  # Considered good if SSIM > 0.70
    },
}
