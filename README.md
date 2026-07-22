# ArchBench

A benchmark for evaluating Large Language Models on Software Architecture tasks.

## Overview

ArchBench provides standardized evaluation for LLMs on four software architecture tasks:

| Task | Description | Primary Metric |
|------|-------------|----------------|
| **ADR** | Architecture Decision Record generation | BERTScore F1 |
| **Traceability** | Architecture-to-code traceability link recovery | F1 Score |
| **Diagram** | Architecture view generation (PlantUML diagrams) | SSIM |
| **Serverless** | Serverless component generation | Test Pass Rate |
| **Dynamic** | Dynamic IoT service generation | CodeBERTScore |

## Installation

```bash
# Basic installation (evaluation only)
pip install -e .

# With inference support (OpenAI, Anthropic, etc.)
pip install -e ".[inference]"

# Full installation (all dependencies)
pip install -e ".[all]"
```

## Try it Out

Run a quick example in under a minute (no PyTorch needed):

```bash
pip install -e ".[eval,inference]"
export OPENAI_API_KEY=your_key_here

# Run ADR inference on 2 samples, auto-downloads dataset, evaluates results
archbench inference --task adr --model gpt-3.5-turbo --output_dir results/ --evaluate --max_instances 2
```

## Quick Start

### 1. Evaluate Predictions

```bash
# Evaluate ADR predictions against ground truth
archbench evaluate \
    --task adr \
    --predictions_path predictions.jsonl \
    --dataset_path ../ArchAI_ADR/data/0_shot.csv \
    --output_dir results/
```

### 2. Run Inference

```bash
# Generate predictions using GPT-4
export OPENAI_API_KEY=your_key_here

archbench inference \
    --task adr \
    --model gpt-4 \
    --dataset_path ../ArchAI_ADR/data/0_shot.csv \
    --output_dir results/ \
    --prompt_style few_shot
```

### 3. LLM-as-a-Judge

Get a qualitative assessment of your predictions using another LLM as a judge. This makes a single API call with aggregated metrics and a few sampled instances, and returns an overall score with strengths, weaknesses, and cited examples.

```bash
# Judge ADR predictions (auto-detects metrics from previous evaluation)
archbench judge \
    --task adr \
    --predictions_path results/predictions.jsonl \
    --judge_model gpt-4

# Control how many instances are sampled into the prompt
archbench judge \
    --task adr \
    --predictions_path results/predictions.jsonl \
    --judge_model gpt-4 \
    --sample_count 10
```

### 4. Validate Submission

```bash
# Check if your predictions file is correctly formatted
archbench validate \
    --task adr \
    --predictions_path predictions.jsonl
```

## Submission Format

Predictions should be in JSONL format with one prediction per line:

```json
{"instance_id": "adr_0000", "model_name_or_path": "gpt-4", "prediction": "We decided to use...", "raw_output": "## Decision\nWe decided to use..."}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | string | Unique identifier matching dataset |
| `model_name_or_path` | string | Name of the model used |
| `prediction` | string | The parsed/extracted prediction |
| `raw_output` | string | Raw model output (for verification) |

### Optional Fields (Recommended)

| Field | Type | Description |
|-------|------|-------------|
| `latency_ms` | float | Response latency in milliseconds |
| `token_usage` | object | Token counts `{prompt_tokens, completion_tokens}` |

## Trajectory Logging

ArchBench logs complete inference trajectories for verification:

```
results/
└── gpt-4_20240215_143022/
    ├── predictions.jsonl          # All predictions
    ├── inference_summary.json     # Run statistics
    └── trajectories/
        ├── all_trajectories.jsonl # All trajectories in one file
        ├── adr_0000.json          # Per-instance trajectory
        ├── adr_0001.json
        └── ...
```

Each trajectory contains:
- Exact prompt sent to the model
- Raw model response
- Parsed output
- Timing and token usage
- Any errors encountered

## Evaluation Metrics

### ADR Task

| Metric | Description |
|--------|-------------|
| ROUGE-1/2/L | N-gram overlap |
| BLEU | Precision-based similarity |
| METEOR | Alignment-based similarity |
| BERTScore P/R/F1 | Semantic similarity using BERT |

### Traceability Task

| Metric | Description |
|--------|-------------|
| Precision | Correct links / Predicted links |
| Recall | Correct links / Actual links |
| F1 | Harmonic mean of P and R |

### Diagram Task

Generated diagrams are rendered from PlantUML and compared against ground truth
diagram images. An optional LLM-as-a-judge scores the 3 C's (Clarity,
Completeness, Consistency).

| Metric | Description |
|--------|-------------|
| SSIM | Structural similarity index |
| PSNR | Peak signal-to-noise ratio |
| RMSE | Root mean squared error |
| SAM | Spectral angle mapper |
| SRE | Signal to reconstruction error |
| UIQ | Universal image quality index |

The diagram task needs image dependencies and the `plantuml` binary:

```bash
pip install -e ".[diagram]"   # opencv, scikit-image, image-similarity-measures
```

Generation, evaluation and the judge run as a single command. Diagrams are
rendered from the generated PlantUML, compared against the ground truth images
(matched by file stem), and judged on the 3 C's. The judge is optional and only
runs when an API key for the judge model is available:

```bash
archbench inference \
    --task diagram \
    --model claude-3-5-sonnet-20240620 \
    --dataset_path generated_summaries.jsonl \
    --ground_truth_dir ground_truth_views/ \
    --output_dir results/ \
    --evaluate \
    --judge
```

Each prediction records both the generated and the ground truth image, so the
steps can also be run separately against an existing predictions file:

```bash
archbench evaluate \
    --task diagram \
    --predictions_path results/<run>/predictions.jsonl \
    --output_dir results/

archbench judge \
    --task diagram \
    --predictions_path results/<run>/predictions.jsonl \
    --judge_model gpt-4o
```

## Python API

```python
from archbench import load_dataset, load_predictions, run_evaluation
from archbench.inference import run_inference

# Load dataset
dataset = load_dataset("adr", dataset_path="data/0_shot.csv")

# Run inference
summary = run_inference(
    task="adr",
    model="gpt-4",
    dataset_path="data/0_shot.csv",
    output_dir="results/",
    prompt_style="few_shot",
)

# Evaluate
report = run_evaluation(
    task="adr",
    predictions_path="results/gpt-4_*/predictions.jsonl",
    dataset_path="data/0_shot.csv",
)

print(f"BERTScore F1: {report['primary_metric_value']:.4f}")
```

## Leaderboard Submission

To submit results to the ArchBench leaderboard:

1. **Run inference** with trajectory logging enabled
2. **Verify** your predictions file format
3. **Create a PR** to the [archbench-results](https://github.com/sa4s-serc/archbench-results) repository with:
   - `predictions.jsonl` - Your model's predictions
   - `trajectories/` - Complete inference trajectories
   - `metadata.yaml` - Submission metadata

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed submission guidelines.
