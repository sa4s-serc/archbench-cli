"""
Grading and metrics computation for ArchBench.

Implements evaluation metrics for each task type:
- ADR: ROUGE, BLEU, METEOR, BERTScore
- Traceability: Precision, Recall, F1
- Serverless: Test pass rates, CodeBLEU
- Dynamic: CodeBERTScore
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from archbench.constants import (
    TASKS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    KEY_DECISION,
    KEY_CONTEXT,
    EvalStatus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Loading of Metrics (to avoid import overhead)
# =============================================================================

_METRICS_CACHE = {}


def _get_rouge():
    if "rouge" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["rouge"] = load("rouge")
        except ImportError:
            logger.warning("evaluate not installed. Install with: pip install archbench[eval]")
            _METRICS_CACHE["rouge"] = None
    return _METRICS_CACHE["rouge"]


def _get_bleu():
    if "bleu" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["bleu"] = load("bleu")
        except ImportError:
            logger.warning("evaluate not installed. Install with: pip install archbench[eval]")
            _METRICS_CACHE["bleu"] = None
    return _METRICS_CACHE["bleu"]


def _get_meteor():
    if "meteor" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["meteor"] = load("meteor")
        except ImportError:
            logger.warning("evaluate not installed. Install with: pip install archbench[eval]")
            _METRICS_CACHE["meteor"] = None
    return _METRICS_CACHE["meteor"]


def _get_bertscore():
    if "bertscore" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["bertscore"] = load("bertscore")
        except ImportError:
            logger.warning("BERTScore not available. Install with: pip install archbench[bertscore]")
            _METRICS_CACHE["bertscore"] = None
    return _METRICS_CACHE["bertscore"]


def _bertscore_available() -> bool:
    """Check if BERTScore is available (torch installed)."""
    try:
        import torch
        return True
    except ImportError:
        return False


# =============================================================================
# ADR Metrics
# =============================================================================

def compute_adr_metrics(
    prediction: str,
    reference: str,
    compute_bertscore: bool = True,
) -> Dict[str, float]:
    """
    Compute all metrics for a single ADR prediction.

    Args:
        prediction: Generated ADR decision text
        reference: Ground truth ADR decision text
        compute_bertscore: Whether to compute BERTScore (slower but more accurate)

    Returns:
        Dictionary of metric scores

    Example:
        >>> scores = compute_adr_metrics(
        ...     prediction="We decided to use PostgreSQL for its reliability.",
        ...     reference="We chose PostgreSQL as our database due to its ACID compliance."
        ... )
        >>> print(scores["bertscore_f1"])
        0.87
    """
    if not prediction or not reference:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "bertscore_p": 0.0,
            "bertscore_r": 0.0,
            "bertscore_f1": 0.0,
        }

    metrics = {}

    # ROUGE scores
    rouge = _get_rouge()
    if rouge is not None:
        try:
            rouge_results = rouge.compute(
                predictions=[prediction],
                references=[reference]
            )
            metrics["rouge1"] = rouge_results["rouge1"]
            metrics["rouge2"] = rouge_results["rouge2"]
            metrics["rougeL"] = rouge_results["rougeL"]
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            metrics["rouge1"] = 0.0
            metrics["rouge2"] = 0.0
            metrics["rougeL"] = 0.0
    else:
        metrics["rouge1"] = None
        metrics["rouge2"] = None
        metrics["rougeL"] = None

    # BLEU score
    bleu = _get_bleu()
    if bleu is not None:
        try:
            bleu_results = bleu.compute(
                predictions=[prediction],
                references=[[reference]]  # BLEU expects list of reference lists
            )
            metrics["bleu"] = bleu_results["bleu"]
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            metrics["bleu"] = 0.0
    else:
        metrics["bleu"] = None

    # METEOR score
    meteor = _get_meteor()
    if meteor is not None:
        try:
            meteor_results = meteor.compute(
                predictions=[prediction],
                references=[reference]
            )
            metrics["meteor"] = meteor_results["meteor"]
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")
            metrics["meteor"] = 0.0
    else:
        metrics["meteor"] = None

    # BERTScore (optional, requires torch)
    if compute_bertscore and _bertscore_available():
        try:
            bertscore = _get_bertscore()
            if bertscore is not None:
                bert_results = bertscore.compute(
                    predictions=[prediction],
                    references=[reference],
                    lang="en"
                )
                metrics["bertscore_p"] = float(np.mean(bert_results["precision"]))
                metrics["bertscore_r"] = float(np.mean(bert_results["recall"]))
                metrics["bertscore_f1"] = float(np.mean(bert_results["f1"]))
            else:
                metrics["bertscore_p"] = None
                metrics["bertscore_r"] = None
                metrics["bertscore_f1"] = None
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            metrics["bertscore_p"] = 0.0
            metrics["bertscore_r"] = 0.0
            metrics["bertscore_f1"] = 0.0
    else:
        if compute_bertscore and not _bertscore_available():
            logger.info("BERTScore skipped (torch not installed). Install with: pip install archbench[bertscore]")
        metrics["bertscore_p"] = None
        metrics["bertscore_r"] = None
        metrics["bertscore_f1"] = None

    return metrics


def compute_adr_metrics_batch(
    predictions: List[str],
    references: List[str],
    compute_bertscore: bool = True,
) -> Dict[str, List[float]]:
    """
    Compute metrics for a batch of ADR predictions (more efficient).

    Args:
        predictions: List of generated ADR decision texts
        references: List of ground truth ADR decision texts
        compute_bertscore: Whether to compute BERTScore

    Returns:
        Dictionary mapping metric names to lists of scores
    """
    n = len(predictions)
    assert len(references) == n, "Predictions and references must have same length"

    # Filter out empty predictions
    valid_pairs = [
        (i, p, r) for i, (p, r) in enumerate(zip(predictions, references))
        if p and r and p.strip() and r.strip()
    ]

    if not valid_pairs:
        return {
            "rouge1": [0.0] * n,
            "rouge2": [0.0] * n,
            "rougeL": [0.0] * n,
            "bleu": [0.0] * n,
            "meteor": [0.0] * n,
            "bertscore_p": [0.0] * n,
            "bertscore_r": [0.0] * n,
            "bertscore_f1": [0.0] * n,
        }

    valid_indices, valid_preds, valid_refs = zip(*valid_pairs)

    # Initialize results
    results = {
        "rouge1": [0.0] * n,
        "rouge2": [0.0] * n,
        "rougeL": [0.0] * n,
        "bleu": [0.0] * n,
        "meteor": [0.0] * n,
        "bertscore_p": [0.0] * n,
        "bertscore_r": [0.0] * n,
        "bertscore_f1": [0.0] * n,
    }

    # Compute ROUGE in batch
    try:
        rouge = _get_rouge()
        for idx, pred, ref in zip(valid_indices, valid_preds, valid_refs):
            rouge_result = rouge.compute(predictions=[pred], references=[ref])
            results["rouge1"][idx] = rouge_result["rouge1"]
            results["rouge2"][idx] = rouge_result["rouge2"]
            results["rougeL"][idx] = rouge_result["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")

    # Compute BLEU
    try:
        bleu = _get_bleu()
        for idx, pred, ref in zip(valid_indices, valid_preds, valid_refs):
            bleu_result = bleu.compute(predictions=[pred], references=[[ref]])
            results["bleu"][idx] = bleu_result["bleu"]
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")

    # Compute METEOR
    try:
        meteor = _get_meteor()
        for idx, pred, ref in zip(valid_indices, valid_preds, valid_refs):
            meteor_result = meteor.compute(predictions=[pred], references=[ref])
            results["meteor"][idx] = meteor_result["meteor"]
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")

    # Compute BERTScore in batch (more efficient, requires torch)
    if compute_bertscore and _bertscore_available():
        try:
            bertscore = _get_bertscore()
            if bertscore is not None:
                bert_results = bertscore.compute(
                    predictions=list(valid_preds),
                    references=list(valid_refs),
                    lang="en"
                )
                for i, idx in enumerate(valid_indices):
                    results["bertscore_p"][idx] = float(bert_results["precision"][i])
                    results["bertscore_r"][idx] = float(bert_results["recall"][i])
                    results["bertscore_f1"][idx] = float(bert_results["f1"][i])
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
    elif compute_bertscore:
        logger.info("BERTScore skipped (torch not installed). Install with: pip install archbench[bertscore]")

    return results


# =============================================================================
# Traceability Metrics
# =============================================================================

def compute_traceability_metrics(
    predicted_links: List[Dict],
    reference_links: List[Dict],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for traceability link recovery.

    Args:
        predicted_links: List of predicted trace links
        reference_links: List of ground truth trace links

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    # Convert to sets of tuples for comparison
    def links_to_set(links):
        return {
            (link.get("doc_sentence") or link.get("doc_sent"), link.get("code_artifact") or link.get("code"))
            for link in links
        }

    pred_set = links_to_set(predicted_links)
    ref_set = links_to_set(reference_links)

    if not ref_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(pred_set & ref_set)

    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# General Metrics Interface
# =============================================================================

def compute_metrics(
    task: str,
    prediction: Any,
    reference: Any,
    **kwargs,
) -> Dict[str, float]:
    """
    Compute metrics for a given task.

    Args:
        task: Task name (adr, traceability, serverless, dynamic)
        prediction: Model prediction
        reference: Ground truth reference
        **kwargs: Additional arguments for specific metrics

    Returns:
        Dictionary of metric scores
    """
    if task == "adr":
        return compute_adr_metrics(
            prediction=str(prediction),
            reference=str(reference),
            compute_bertscore=kwargs.get("compute_bertscore", True),
        )
    elif task == "traceability":
        # Parse JSON if string
        if isinstance(prediction, str):
            try:
                prediction = json.loads(prediction)
            except json.JSONDecodeError:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return compute_traceability_metrics(
            predicted_links=prediction,
            reference_links=reference,
        )
    elif task == "diagram":
        from archbench.tasks.diagram import grading as diagram_grading
        # For diagram, prediction/reference are generated/ground-truth image paths
        return diagram_grading.compute_diagram_metrics(
            generated_image_path=str(prediction),
            ground_truth_image_path=str(reference),
        )
    elif task == "serverless":
        # TODO: Implement serverless metrics (CodeBLEU, test pass rates)
        logger.warning("Serverless metrics not yet implemented")
        return {}
    elif task == "dynamic":
        # TODO: Implement dynamic IoT metrics (CodeBERTScore)
        logger.warning("Dynamic IoT metrics not yet implemented")
        return {}
    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_metrics(
    instance_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate per-instance metrics into summary statistics.

    Args:
        instance_metrics: List of metric dictionaries for each instance

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not instance_metrics:
        return {}

    # Get all metric names
    metric_names = set()
    for m in instance_metrics:
        metric_names.update(m.keys())

    aggregated = {}
    for metric in metric_names:
        values = [m.get(metric, 0.0) for m in instance_metrics if m.get(metric) is not None]
        if values:
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
            aggregated[f"{metric}_min"] = float(np.min(values))
            aggregated[f"{metric}_max"] = float(np.max(values))

    return aggregated


def get_resolution_status(
    metrics: Dict[str, float],
    task: str,
) -> str:
    """
    Determine if an instance is "resolved" based on metrics.

    Args:
        metrics: Dictionary of metric scores
        task: Task name

    Returns:
        Resolution status (success, partial, no)
    """
    from archbench.constants import METRIC_THRESHOLDS

    thresholds = METRIC_THRESHOLDS.get(task, {})

    for metric, threshold in thresholds.items():
        if metric in metrics and metrics[metric] >= threshold:
            return EvalStatus.SUCCESS.value

    return EvalStatus.ERROR.value
