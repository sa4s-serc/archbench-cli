#!/usr/bin/env python3
"""
Main evaluation script for ArchBench.

Usage:
    python -m archbench.harness.run_evaluation \
        --task adr \
        --predictions_path predictions.jsonl \
        --dataset_path data/0_shot.csv \
        --output_dir results/

This script:
1. Loads the dataset and predictions
2. Validates predictions format
3. Computes metrics for each instance
4. Generates a comprehensive evaluation report
"""

import json
import logging
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from archbench.constants import (
    TASKS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    KEY_DECISION,
    KEY_CONTEXT,
    LOG_REPORT,
    LOG_METRICS,
    EvalStatus,
)
from archbench.harness.utils import (
    load_dataset,
    load_predictions,
    get_predictions_from_file,
    validate_predictions,
    save_report,
)
from archbench.tasks.adr import dataset as adr_dataset
from archbench.tasks.adr import grading as adr_grading
from archbench.harness.grading import (
    aggregate_metrics,
    get_resolution_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Main Evaluation Function
# =============================================================================

def run_evaluation(
    task: str,
    predictions_path: str,
    dataset_path: Optional[str] = None,
    output_dir: str = "results",
    run_id: Optional[str] = None,
    compute_bertscore: bool = True,
    instance_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run evaluation for a given task.

    Args:
        task: Task name (adr, traceability, serverless, dynamic)
        predictions_path: Path to predictions file (JSONL)
        dataset_path: Path to dataset file (CSV/JSON/JSONL) or None for HuggingFace
        output_dir: Directory to save results
        run_id: Unique identifier for this evaluation run
        compute_bertscore: Whether to compute BERTScore (slower but more accurate)
        instance_ids: Optional list of specific instance IDs to evaluate

    Returns:
        Evaluation report dictionary
    """
    start_time = time.time()

    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Starting evaluation run: {run_id}")
    logger.info(f"Task: {task}")
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Dataset: {dataset_path or 'HuggingFace'}")

    # Validate task
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASKS.keys())}")

    # Load dataset
    logger.info("Loading dataset...")
    if task == "adr":
        dataset = adr_dataset.load_dataset(dataset_path=dataset_path, instance_ids=instance_ids)
    elif task == "traceability":
        from archbench.tasks.traceability import dataset as trace_dataset
        dataset = trace_dataset.load_dataset(dataset_path=dataset_path, task_type="sad-code")
    elif task == "diagram":
        from archbench.tasks.diagram import dataset as diagram_dataset
        dataset = diagram_dataset.load_dataset(
            dataset_path=dataset_path,
            instance_ids=instance_ids,
            predictions_path=predictions_path,
        )
    else:
        # TODO: Add other task loaders
        dataset = load_dataset(task, dataset_path=dataset_path, instance_ids=instance_ids)

    # Create instance lookup
    dataset_dict = {d[KEY_INSTANCE_ID]: d for d in dataset}
    logger.info(f"Loaded {len(dataset)} instances")

    # Load predictions
    logger.info("Loading predictions...")
    predictions = load_predictions(predictions_path)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Validate predictions
    logger.info("Validating predictions...")
    validation_result = validate_predictions(predictions, task)

    # Filter to evaluated instances
    if instance_ids:
        eval_ids = set(instance_ids)
    else:
        eval_ids = set(dataset_dict.keys())

    # Track results
    instance_results = []
    completed_ids = []
    error_ids = []
    missing_ids = []

    # Evaluate each instance
    logger.info("Computing metrics...")

    if task == "adr":
        # Batch processing for ADR (more efficient)
        instance_results = evaluate_adr_batch(
            dataset=dataset,
            predictions=predictions,
            compute_bertscore=compute_bertscore,
        )
        for result in instance_results:
            if result["status"] == EvalStatus.SUCCESS.value:
                completed_ids.append(result[KEY_INSTANCE_ID])
            elif result["status"] == EvalStatus.MISSING_PREDICTION.value:
                missing_ids.append(result[KEY_INSTANCE_ID])
            else:
                error_ids.append(result[KEY_INSTANCE_ID])
    elif task == "traceability":
        # Batch processing for traceability
        instance_results = evaluate_traceability_batch(
            dataset=dataset,
            predictions=predictions,
        )
        for result in instance_results:
            if result["status"] == EvalStatus.SUCCESS.value:
                completed_ids.append(result[KEY_INSTANCE_ID])
            elif result["status"] == EvalStatus.MISSING_PREDICTION.value:
                missing_ids.append(result[KEY_INSTANCE_ID])
            else:
                error_ids.append(result[KEY_INSTANCE_ID])
    elif task == "diagram":
        # Batch processing for diagram (image similarity)
        instance_results = evaluate_diagram_batch(
            dataset=dataset,
            predictions=predictions,
        )
        for result in instance_results:
            if result["status"] == EvalStatus.SUCCESS.value:
                completed_ids.append(result[KEY_INSTANCE_ID])
            elif result["status"] == EvalStatus.MISSING_PREDICTION.value:
                missing_ids.append(result[KEY_INSTANCE_ID])
            else:
                error_ids.append(result[KEY_INSTANCE_ID])
    else:
        # Sequential processing for other tasks
        for instance in tqdm(dataset, desc="Evaluating"):
            instance_id = instance[KEY_INSTANCE_ID]

            if instance_id not in predictions:
                missing_ids.append(instance_id)
                instance_results.append({
                    KEY_INSTANCE_ID: instance_id,
                    "status": EvalStatus.MISSING_PREDICTION.value,
                    "metrics": {},
                })
                continue

            pred = predictions[instance_id]
            prediction = pred.get(KEY_PREDICTION, "")

            if not prediction or not prediction.strip():
                error_ids.append(instance_id)
                instance_results.append({
                    KEY_INSTANCE_ID: instance_id,
                    "status": EvalStatus.INVALID_FORMAT.value,
                    "metrics": {},
                })
                continue

            # Get reference
            reference = get_reference_for_task(task, instance)

            # Compute metrics
            try:
                metrics = compute_metrics(
                    task=task,
                    prediction=prediction,
                    reference=reference,
                    compute_bertscore=compute_bertscore,
                )
                completed_ids.append(instance_id)
                instance_results.append({
                    KEY_INSTANCE_ID: instance_id,
                    "status": EvalStatus.SUCCESS.value,
                    "metrics": metrics,
                    "model": pred.get(KEY_MODEL, "unknown"),
                })
            except Exception as e:
                logger.error(f"Error evaluating {instance_id}: {e}")
                error_ids.append(instance_id)
                instance_results.append({
                    KEY_INSTANCE_ID: instance_id,
                    "status": EvalStatus.ERROR.value,
                    "error": str(e),
                    "metrics": {},
                })

    # Aggregate metrics
    logger.info("Aggregating results...")
    valid_metrics = [r["metrics"] for r in instance_results if r["metrics"]]
    aggregated = aggregate_metrics(valid_metrics)

    # Compute elapsed time
    elapsed_time = time.time() - start_time

    # Build report
    report = make_report(
        task=task,
        run_id=run_id,
        predictions_path=predictions_path,
        dataset_path=dataset_path,
        total_instances=len(dataset),
        completed_ids=completed_ids,
        error_ids=error_ids,
        missing_ids=missing_ids,
        instance_results=instance_results,
        aggregated_metrics=aggregated,
        elapsed_time=elapsed_time,
        validation_result=validation_result,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save main report
    report_file = output_path / f"{run_id}_{task}_{LOG_REPORT}"
    save_report(report, str(report_file))

    # Save per-instance metrics
    metrics_file = output_path / f"{run_id}_{task}_{LOG_METRICS}"
    with open(metrics_file, "w") as f:
        for result in instance_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved per-instance metrics to {metrics_file}")

    # Print summary
    print_summary(report)

    return report


def evaluate_adr_batch(
    dataset: List[Dict],
    predictions: Dict[str, Dict],
    compute_bertscore: bool = True,
) -> List[Dict]:
    """
    Evaluate ADR task in batch mode for efficiency.
    """
    results = []
    valid_indices = []
    valid_preds = []
    valid_refs = []

    for i, instance in enumerate(dataset):
        instance_id = instance[KEY_INSTANCE_ID]

        if instance_id not in predictions:
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.MISSING_PREDICTION.value,
                "metrics": {},
            })
            continue

        pred = predictions[instance_id]
        prediction = pred.get(KEY_PREDICTION, "")
        reference = instance.get(KEY_DECISION, "")

        if not prediction or not prediction.strip():
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.INVALID_FORMAT.value,
                "metrics": {},
            })
            continue

        valid_indices.append(len(results))
        valid_preds.append(prediction)
        valid_refs.append(reference)
        results.append({
            KEY_INSTANCE_ID: instance_id,
            "status": EvalStatus.SUCCESS.value,
            "model": pred.get(KEY_MODEL, "unknown"),
            "metrics": {},  # Will be filled in
        })

    # Compute metrics in batch
    if valid_preds:
        logger.info(f"Computing metrics for {len(valid_preds)} valid predictions...")
        batch_metrics = adr_grading.compute_adr_metrics_batch(
            predictions=valid_preds,
            references=valid_refs,
            compute_bertscore=compute_bertscore,
        )

        # Assign metrics back to results
        for i, idx in enumerate(valid_indices):
            results[idx]["metrics"] = {
                metric: values[i]
                for metric, values in batch_metrics.items()
            }

    return results


def parse_traceability_prediction(prediction: Any) -> List[Tuple]:
    """
    Parse traceability prediction into list of tuples.
    Handles both new format (list) and old format (JSON string).
    """
    from archbench.tasks.traceability import dataset as trace_dataset

    if isinstance(prediction, list):
        # New format: already a list of tuples/lists
        return [tuple(item) if isinstance(item, list) else item for item in prediction]
    elif isinstance(prediction, str):
        # Old format: JSON string (double-encoded)
        import json
        try:
            parsed = json.loads(prediction)
            if isinstance(parsed, list):
                return [tuple(item) for item in parsed]
            else:
                # Raw LLM output, use extract_prediction
                return trace_dataset.extract_prediction(prediction, task_type="sad-code")
        except json.JSONDecodeError:
            # Raw LLM output
            return trace_dataset.extract_prediction(prediction, task_type="sad-code")
    else:
        raise ValueError(f"Unknown prediction type: {type(prediction)}")


def evaluate_traceability_batch(
    dataset: List[Dict],
    predictions: Dict[str, Dict],
) -> List[Dict]:
    """
    Evaluate traceability task in batch mode.
    """
    from archbench.tasks.traceability import grading as trace_grading

    results = []

    for instance in dataset:
        instance_id = instance[KEY_INSTANCE_ID]

        if instance_id not in predictions:
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.MISSING_PREDICTION.value,
                "metrics": {},
            })
            continue

        pred = predictions[instance_id]
        prediction = pred.get(KEY_PREDICTION, "")
        reference_links = instance.get("goldstandard", [])

        if not prediction:
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.INVALID_FORMAT.value,
                "metrics": {},
            })
            continue

        # Parse prediction
        try:
            predicted_links = parse_traceability_prediction(prediction)
        except Exception as e:
            logger.warning(f"Failed to parse prediction for {instance_id}: {e}")
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.ERROR.value,
                "metrics": {},
            })
            continue

        # Compute metrics
        metrics = trace_grading.compute_traceability_metrics(
            predicted_links=predicted_links,
            reference_links=reference_links,
        )

        results.append({
            KEY_INSTANCE_ID: instance_id,
            "status": EvalStatus.SUCCESS.value,
            "model": pred.get(KEY_MODEL, "unknown"),
            "metrics": metrics,
        })

    return results


def evaluate_diagram_batch(
    dataset: List[Dict],
    predictions: Dict[str, Dict],
) -> List[Dict]:
    """
    Evaluate the architecture view generation task in batch mode.

    Each prediction is expected to carry a ``generated_image`` path (produced
    during inference by rendering the PlantUML code). The generated image is
    compared against the instance's ground truth image via image similarity
    metrics, falling back to the ``ground_truth_image`` recorded on the
    prediction when the dataset instance does not carry one.
    """
    from archbench.constants import KEY_GROUND_TRUTH_IMAGE, KEY_GENERATED_IMAGE
    from archbench.tasks.diagram import grading as diagram_grading

    results = []
    valid_indices = []
    valid_gen_paths = []
    valid_gt_paths = []

    for instance in dataset:
        instance_id = instance[KEY_INSTANCE_ID]

        if instance_id not in predictions:
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.MISSING_PREDICTION.value,
                "metrics": {},
            })
            continue

        pred = predictions[instance_id]
        generated_image = pred.get(KEY_GENERATED_IMAGE, "")
        ground_truth_image = (
            instance.get(KEY_GROUND_TRUTH_IMAGE) or pred.get(KEY_GROUND_TRUTH_IMAGE) or ""
        )

        if not generated_image or not ground_truth_image:
            results.append({
                KEY_INSTANCE_ID: instance_id,
                "status": EvalStatus.INVALID_FORMAT.value,
                "metrics": {},
            })
            continue

        valid_indices.append(len(results))
        valid_gen_paths.append(generated_image)
        valid_gt_paths.append(ground_truth_image)
        results.append({
            KEY_INSTANCE_ID: instance_id,
            "status": EvalStatus.SUCCESS.value,
            "model": pred.get(KEY_MODEL, "unknown"),
            "metrics": {},  # Will be filled in
        })

    # Compute metrics in batch
    if valid_gen_paths:
        logger.info(f"Computing image metrics for {len(valid_gen_paths)} valid predictions...")
        batch_metrics = diagram_grading.compute_diagram_metrics_batch(
            generated_image_paths=valid_gen_paths,
            ground_truth_image_paths=valid_gt_paths,
        )
        for i, idx in enumerate(valid_indices):
            results[idx]["metrics"] = {
                metric: values[i]
                for metric, values in batch_metrics.items()
            }

    return results


def get_reference_for_task(task: str, instance: Dict) -> Any:
    """Get the reference/ground truth for a task instance."""
    if task == "adr":
        return instance.get(KEY_DECISION, "")
    elif task == "traceability":
        return instance.get("trace_links", [])
    elif task == "diagram":
        from archbench.constants import KEY_GROUND_TRUTH_IMAGE
        return instance.get(KEY_GROUND_TRUTH_IMAGE, "")
    elif task == "serverless":
        return instance.get("reference_function", "")
    elif task == "dynamic":
        return instance.get("reference_service", "")
    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# Report Generation
# =============================================================================

def make_report(
    task: str,
    run_id: str,
    predictions_path: str,
    dataset_path: Optional[str],
    total_instances: int,
    completed_ids: List[str],
    error_ids: List[str],
    missing_ids: List[str],
    instance_results: List[Dict],
    aggregated_metrics: Dict[str, float],
    elapsed_time: float,
    validation_result: Dict,
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    Generate a structured evaluation report.
    """
    report = {
        # Metadata
        "run_id": run_id,
        "task": task,
        "task_name": TASKS[task]["name"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_seconds": elapsed_time,

        # Input files
        "predictions_path": predictions_path,
        "dataset_path": dataset_path,

        # Summary statistics
        "total_instances": total_instances,
        "completed_instances": len(completed_ids),
        "error_instances": len(error_ids),
        "missing_instances": len(missing_ids),

        # Completion rate
        "completion_rate": len(completed_ids) / total_instances if total_instances > 0 else 0,

        # ID lists
        "completed_ids": sorted(completed_ids),
        "error_ids": sorted(error_ids),
        "missing_ids": sorted(missing_ids),

        # Aggregated metrics
        "metrics": aggregated_metrics,

        # Primary metric (for leaderboard)
        "primary_metric": TASKS[task]["primary_metric"],
        "primary_metric_value": aggregated_metrics.get(
            f"{TASKS[task]['primary_metric']}_mean", 0.0
        ),

        # Validation results
        "validation": validation_result,

        # Schema version for future compatibility
        "schema_version": 1,
    }

    return report


def print_summary(report: Dict) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {report['task_name']}")
    print("=" * 60)
    print(f"Run ID: {report['run_id']}")
    print(f"Total instances: {report['total_instances']}")
    print(f"Completed: {report['completed_instances']}")
    print(f"Errors: {report['error_instances']}")
    print(f"Missing: {report['missing_instances']}")
    print(f"Completion rate: {report['completion_rate']:.1%}")
    print("-" * 60)
    print("METRICS:")
    for metric, value in sorted(report["metrics"].items()):
        if "_mean" in metric:
            print(f"  {metric.replace('_mean', '')}: {value:.4f}")
    print("-" * 60)
    print(f"Primary metric ({report['primary_metric']}): {report['primary_metric_value']:.4f}")
    print(f"Elapsed time: {report['elapsed_time_seconds']:.1f}s")
    print("=" * 60 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = ArgumentParser(
        description="Run ArchBench evaluation harness",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        choices=list(TASKS.keys()),
        help="Task to evaluate",
    )
    parser.add_argument(
        "-p", "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions file (JSONL or JSON)",
    )
    parser.add_argument(
        "-d", "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset file (CSV/JSON/JSONL). If not provided, loads from HuggingFace.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique identifier for this run (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no_bertscore",
        action="store_true",
        help="Skip BERTScore computation (faster but less accurate)",
    )
    parser.add_argument(
        "-i", "--instance_ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific instance IDs to evaluate (space-separated)",
    )

    args = parser.parse_args()

    run_evaluation(
        task=args.task,
        predictions_path=args.predictions_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        compute_bertscore=not args.no_bertscore,
        instance_ids=args.instance_ids,
    )


if __name__ == "__main__":
    main()
