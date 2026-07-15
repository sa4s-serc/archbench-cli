#!/usr/bin/env python3
"""
Command-line interface for ArchBench.

Usage:
    archbench evaluate --task adr --predictions_path preds.jsonl --dataset_path data.csv
    archbench inference --task adr --model gpt-4 --dataset_path data.csv
    archbench validate --task adr --predictions_path preds.jsonl
    archbench download --task adr --output_dir datasets/
"""

import argparse
import sys

from archbench import __version__
from archbench.constants import TASKS


def main():
    parser = argparse.ArgumentParser(
        description="ArchBench - A benchmark for evaluating LLMs on software architecture tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  archbench evaluate --task adr --predictions_path predictions.jsonl --dataset_path data/0_shot.csv
  archbench inference --task adr --model gpt-4 --dataset_path data/0_shot.csv --output_dir results/
  archbench judge --task adr --predictions_path predictions.jsonl --judge_model gpt-4
  archbench validate --task adr --predictions_path predictions.jsonl

Available tasks:
  adr           - Architecture Decision Record generation
  traceability  - Architecture traceability link recovery
  diagram       - Architecture view generation (PlantUML diagrams)
  serverless    - Serverless component generation
  dynamic       - Dynamic IoT service generation

For more information, visit: https://github.com/sa4s-serc/archbench
        """,
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"archbench {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate predictions against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument("-t", "--task", required=True, choices=list(TASKS.keys()))
    eval_parser.add_argument("-p", "--predictions_path", required=True, help="Path to predictions file")
    eval_parser.add_argument("-d", "--dataset_path", default=None, help="Path to dataset file (auto-downloads if not provided)")
    eval_parser.add_argument("-o", "--output_dir", default="results", help="Output directory")
    eval_parser.add_argument("--run_id", default=None, help="Unique run identifier")
    eval_parser.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore computation")

    # Inference command
    infer_parser = subparsers.add_parser(
        "inference",
        help="Run inference using an LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    infer_parser.add_argument("-t", "--task", required=True, choices=list(TASKS.keys()))
    infer_parser.add_argument("-m", "--model", required=True, help="Model name (e.g., gpt-4)")
    infer_parser.add_argument("-d", "--dataset_path", default=None, help="Path to dataset file (auto-downloads if not provided)")
    infer_parser.add_argument("-o", "--output_dir", default="results", help="Output directory")
    infer_parser.add_argument("--prompt_style", default="zero_shot", choices=["zero_shot", "few_shot"])
    infer_parser.add_argument("--max_tokens", type=int, default=1024)
    infer_parser.add_argument("--temperature", type=float, default=0.2)
    infer_parser.add_argument("--run_id", default=None)
    infer_parser.add_argument("--resume_from", default=None, help="Resume from existing predictions")
    infer_parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to process (for testing)")
    infer_parser.add_argument("--evaluate", action="store_true", help="Automatically run evaluation after inference")
    infer_parser.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama server URL (for ollama/ models)")

    # Validate command
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate prediction file format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    val_parser.add_argument("-t", "--task", required=True, choices=list(TASKS.keys()))
    val_parser.add_argument("-p", "--predictions_path", required=True, help="Path to predictions file")

    # Judge command (LLM-as-a-judge evaluation)
    judge_parser = subparsers.add_parser(
        "judge",
        help="Run LLM-as-a-judge evaluation on predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    judge_parser.add_argument("-t", "--task", required=True, choices=list(TASKS.keys()))
    judge_parser.add_argument("-p", "--predictions_path", required=True, help="Path to predictions JSONL")
    judge_parser.add_argument("-d", "--dataset_path", default=None, help="Path to dataset file (auto-downloads if not provided)")
    judge_parser.add_argument("-j", "--judge_model", default="gpt-4", help="Model to use as judge")
    judge_parser.add_argument("-o", "--output_dir", default="results", help="Output directory")
    judge_parser.add_argument("--sample_count", type=int, default=5, help="Number of instances to sample for the judge prompt")
    judge_parser.add_argument("--report_path", default=None, help="Path to evaluation report.json (auto-detected if not provided)")
    judge_parser.add_argument("--metrics_path", default=None, help="Path to per-instance metrics JSONL (auto-detected if not provided)")
    judge_parser.add_argument("--run_id", default=None, help="Unique run identifier")
    judge_parser.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama server URL (for ollama/ models)")

    # Download command (placeholder)
    dl_parser = subparsers.add_parser(
        "download",
        help="Download ArchBench datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dl_parser.add_argument("-t", "--task", default="all", help="Task to download (or 'all')")
    dl_parser.add_argument("-o", "--output_dir", default="datasets", help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "evaluate":
        from archbench.harness.run_evaluation import run_evaluation
        run_evaluation(
            task=args.task,
            predictions_path=args.predictions_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            run_id=args.run_id,
            compute_bertscore=not args.no_bertscore,
        )

    elif args.command == "inference":
        from archbench.inference.run_inference import run_inference

        # Auto-adjust max_tokens for different tasks if not explicitly set
        max_tokens = args.max_tokens
        if args.max_tokens == 1024:  # Default value
            if args.task == "traceability":
                max_tokens = 4096  # Traceability needs more tokens for large link lists (teammates has 198 sentences)

        summary = run_inference(
            task=args.task,
            model=args.model,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            prompt_style=args.prompt_style,
            max_tokens=max_tokens,
            temperature=args.temperature,
            run_id=args.run_id,
            resume_from=args.resume_from,
            limit=args.limit,
            ollama_host=args.ollama_host,
        )

        # Run evaluation if requested
        if args.evaluate:
            print("\n" + "="*60)
            print("Running evaluation...")
            print("="*60 + "\n")
            from archbench.harness.run_evaluation import run_evaluation
            eval_report = run_evaluation(
                task=args.task,
                predictions_path=summary['predictions_file'],
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                compute_bertscore=(args.task == "adr"),  # Only for ADR
            )
            print(f"\n{'='*60}")
            print("Evaluation complete!")
            print(f"{'='*60}")
            print(f"Results saved to: {args.output_dir}")
            print(f"\nSummary:")
            print(f"  Evaluated: {eval_report['completed_instances']}/{eval_report['total_instances']} instances")
            print(f"  Primary metric ({eval_report['primary_metric']}): {eval_report['primary_metric_value']:.4f}")
            print(f"\nMetrics:")
            for metric, value in eval_report['metrics'].items():
                if metric.endswith('_mean'):
                    print(f"  {metric.replace('_mean', '')}: {value:.4f}")
            print(f"{'='*60}\n")

    elif args.command == "judge":
        from archbench.harness.llm_judge import run_llm_judge
        report = run_llm_judge(
            task=args.task,
            predictions_path=args.predictions_path,
            dataset_path=args.dataset_path,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
            sample_count=args.sample_count,
            report_path=args.report_path,
            instance_metrics_path=args.metrics_path,
            run_id=args.run_id,
            ollama_host=args.ollama_host,
        )

    elif args.command == "validate":
        from archbench.harness.utils import load_predictions, validate_predictions
        predictions = load_predictions(args.predictions_path)
        result = validate_predictions(predictions, args.task)
        print(f"\nValidation Results:")
        print(f"  Valid predictions: {len(result['valid'])}")
        print(f"  Invalid predictions: {len(result['invalid'])}")
        if result['errors']:
            print(f"\nErrors:")
            for error in result['errors'][:10]:
                print(f"  - {error}")
            if len(result['errors']) > 10:
                print(f"  ... and {len(result['errors']) - 10} more")

    elif args.command == "download":
        from pathlib import Path
        output_dir = Path(args.output_dir)

        if args.task == "adr" or args.task == "all":
            from archbench.tasks.adr import dataset as adr_dataset
            print(f"Downloading ADR dataset...")
            adr_path = adr_dataset.download_from_github(
                output_dir=str(output_dir / "adr"),
                force=True
            )
            print(f"✓ ADR dataset saved to: {adr_path}")

        if args.task == "all":
            print("\nOther datasets (traceability, serverless, dynamic) will be added soon.")

        print(f"\nAll requested datasets downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
