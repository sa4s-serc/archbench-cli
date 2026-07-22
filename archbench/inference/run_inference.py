#!/usr/bin/env python3
"""
Inference script for ArchBench with trajectory logging.

This script runs LLM inference on ArchBench tasks and logs:
1. The exact prompt sent to the model
2. Raw model response
3. Parsed/extracted output
4. Timing and token usage information
5. Any errors encountered

The trajectory logs enable verification of submissions and reproducibility.

Usage:
    python -m archbench.inference.run_inference \
        --task adr \
        --model gpt-4 \
        --dataset_path data/0_shot.csv \
        --output_dir results/ \
        --prompt_style few_shot

Supported models:
    - OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo, text-davinci-003
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Local: Any HuggingFace model (requires transformers)
"""

import json
import logging
import os
import time
import traceback
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

from archbench.constants import (
    TASKS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    KEY_RAW_OUTPUT,
    KEY_CONTEXT,
    LOG_INFERENCE_DIR,
    LOG_TRAJECTORY,
    LOG_PREDICTIONS,
)
from archbench.harness.utils import (
    load_dataset,
    save_predictions,
)
from archbench.tasks.adr import dataset as adr_dataset
from archbench.tasks.adr import prompts as adr_prompts

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Trajectory Logging
# =============================================================================

@dataclass
class TrajectoryStep:
    """A single step in the inference trajectory."""
    step_type: str              # "prompt", "api_call", "response", "parse", "error"
    timestamp: str
    content: Any
    metadata: Optional[Dict] = None


@dataclass
class InferenceTrajectory:
    """Complete trajectory for a single instance."""
    instance_id: str
    model: str
    prompt_style: str
    start_time: str
    end_time: Optional[str] = None
    steps: List[Dict] = None
    final_prediction: Optional[Any] = None  # Can be str (ADR) or list (traceability)
    raw_output: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    token_usage: Optional[Dict] = None
    latency_ms: Optional[float] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []

    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict] = None):
        step = TrajectoryStep(
            step_type=step_type,
            timestamp=datetime.now().isoformat(),
            content=content,
            metadata=metadata,
        )
        self.steps.append(asdict(step))

    def to_dict(self) -> Dict:
        return asdict(self)


class TrajectoryLogger:
    """Logger for inference trajectories."""

    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = Path(output_dir) / run_id / "trajectories"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories = []

    def log_trajectory(self, trajectory: InferenceTrajectory):
        """Log a single trajectory."""
        self.trajectories.append(trajectory.to_dict())

        # Also write to individual file for debugging
        traj_file = self.output_dir / f"{trajectory.instance_id}.json"
        with open(traj_file, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2, ensure_ascii=False)

    def save_all(self, filename: str = "all_trajectories.jsonl"):
        """Save all trajectories to a single JSONL file."""
        output_path = self.output_dir.parent / filename
        with open(output_path, "w") as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.trajectories)} trajectories to {output_path}")


# =============================================================================
# Model Providers
# =============================================================================

class ModelProvider:
    """Base class for model providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def generate(self, messages: List[Dict], **kwargs) -> Tuple[str, Dict]:
        """Generate completion and return (response_text, metadata)."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI API provider using httpx directly (bypasses openai library bugs)."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        import httpx
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = httpx.Client(timeout=60)
        self.base_url = "https://api.openai.com/v1"

    @retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs,
    ) -> Tuple[str, Dict]:
        start_time = time.time()

        print(f"[DEBUG] Calling OpenAI API with model: {self.model_name}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"[DEBUG] API ERROR: {type(e).__name__}: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000
        print(f"[DEBUG] API call successful, latency: {latency_ms:.0f}ms")

        text = result["choices"][0]["message"]["content"]
        metadata = {
            "model": result["model"],
            "usage": {
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
            },
            "finish_reason": result["choices"][0]["finish_reason"],
            "latency_ms": latency_ms,
        }

        return text, metadata


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs,
    ) -> Tuple[str, Dict]:
        start_time = time.time()

        # Extract system message
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=user_messages,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000

        text = response.content[0].text
        metadata = {
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "stop_reason": response.stop_reason,
            "latency_ms": latency_ms,
        }

        return text, metadata


class OllamaProvider(ModelProvider):
    """Ollama local inference provider via OpenAI-compatible API."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        import httpx
        self.client = httpx.Client(timeout=120)
        self.base_url = host.rstrip("/")
        self.ollama_model = model_name[len("ollama/"):] if model_name.startswith("ollama/") else model_name

    def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs,
    ) -> Tuple[str, Dict]:
        start_time = time.time()

        print(f"[DEBUG] Calling Ollama with model: {self.ollama_model}")

        data = {
            "model": self.ollama_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            response = self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"[DEBUG] Ollama ERROR: {type(e).__name__}: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000
        print(f"[DEBUG] Ollama call successful, latency: {latency_ms:.0f}ms")

        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        metadata = {
            "model": result.get("model", self.ollama_model),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "finish_reason": result["choices"][0].get("finish_reason"),
            "latency_ms": latency_ms,
        }

        return text, metadata


def get_provider(model_name: str, ollama_host: str = "http://localhost:11434") -> ModelProvider:
    """Get the appropriate provider for a model."""
    model_lower = model_name.lower()

    if model_lower.startswith("ollama/"):
        return OllamaProvider(model_name, host=ollama_host)
    elif any(x in model_lower for x in ["gpt-4", "gpt-3.5", "davinci", "text-"]):
        return OpenAIProvider(model_name)
    elif any(x in model_lower for x in ["claude"]):
        return AnthropicProvider(model_name)
    else:
        raise ValueError(f"Unknown model provider for: {model_name}")


# =============================================================================
# Response Parsing
# =============================================================================

def extract_response(task: str, response: str) -> Any:
    """Extract the relevant output from model response based on task."""
    if task == "adr":
        return adr_dataset.extract_prediction(response)
    elif task == "traceability":
        from archbench.tasks.traceability import dataset as trace_dataset
        return trace_dataset.extract_prediction(response, task_type="sad-code")
    elif task == "diagram":
        from archbench.tasks.diagram import dataset as diagram_dataset
        return diagram_dataset.extract_prediction(response)
    else:
        # TODO: Add other task extraction
        return response


def render_plantuml(puml_code: str, instance_id: str, output_dir: Path) -> Optional[str]:
    """
    Render PlantUML code to a diagram image using the local ``plantuml`` binary.

    Args:
        puml_code: The PlantUML source to render
        instance_id: Used to name the generated image and .puml file
        output_dir: Directory to write the rendered image into

    Returns:
        Path to the generated image, or None if rendering failed (e.g. plantuml
        not installed or the code did not compile).
    """
    import glob
    import shutil
    import subprocess
    import tempfile

    if not puml_code.strip():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    puml_dir = output_dir / "plantuml_code"
    puml_dir.mkdir(parents=True, exist_ok=True)
    puml_file = puml_dir / f"{instance_id}.puml"
    puml_file.write_text(puml_code, encoding="utf-8")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["plantuml", "-o", temp_dir, str(puml_file)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.warning(f"PlantUML rendering failed for {instance_id}: {result.stderr.strip()}")
                return None

            generated_files = glob.glob(os.path.join(temp_dir, "*"))
            if not generated_files:
                logger.warning(f"PlantUML produced no output for {instance_id}")
                return None

            generated_file = generated_files[0]
            ext = os.path.splitext(generated_file)[1]
            output_path = output_dir / f"{instance_id}{ext}"
            shutil.move(generated_file, str(output_path))
            return str(output_path)
    except FileNotFoundError:
        logger.warning("plantuml binary not found; skipping render. Install PlantUML to enable image evaluation.")
        return None
    except Exception as e:
        logger.warning(f"PlantUML rendering error for {instance_id}: {e}")
        return None


# =============================================================================
# Main Inference Function
# =============================================================================

def run_inference_single(
    task: str,
    instance: Dict,
    provider: ModelProvider,
    prompt_style: str = "zero_shot",
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> InferenceTrajectory:
    """
    Run inference on a single instance with full trajectory logging.

    Args:
        task: Task name
        instance: Dataset instance
        provider: Model provider
        prompt_style: Prompt style (zero_shot, few_shot)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        InferenceTrajectory with all logged steps
    """
    instance_id = instance[KEY_INSTANCE_ID]

    trajectory = InferenceTrajectory(
        instance_id=instance_id,
        model=provider.model_name,
        prompt_style=prompt_style,
        start_time=datetime.now().isoformat(),
    )

    try:
        # Step 1: Create prompt
        if task == "adr":
            messages = adr_prompts.create_chat_messages(
                context=instance[KEY_CONTEXT],
                prompt_style=prompt_style,
            )
        elif task == "traceability":
            from archbench.tasks.traceability import prompts as trace_prompts
            messages = trace_prompts.create_chat_messages(
                sentences=instance["sentences"],
                task_type="sad-code",
                code_files=instance.get("available_targets", []),  # Provide actual code files!
                prompt_style=prompt_style,
            )
        elif task == "diagram":
            from archbench.tasks.diagram import prompts as diagram_prompts
            from archbench.constants import KEY_SUMMARY, KEY_CONCERN, KEY_BEHAVIOR
            messages = diagram_prompts.create_chat_messages(
                summary=instance[KEY_SUMMARY],
                concern=instance.get(KEY_CONCERN, "general"),
                behavior=instance.get(KEY_BEHAVIOR, "static"),
                repo_name=instance_id,
                prompt_style=prompt_style,
            )
        else:
            # TODO: Add other task prompts
            raise NotImplementedError(f"Task {task} not yet implemented")

        trajectory.add_step(
            step_type="prompt",
            content=messages,
            metadata={"num_messages": len(messages)},
        )

        # Step 2: Call API
        trajectory.add_step(
            step_type="api_call",
            content={"model": provider.model_name, "max_tokens": max_tokens, "temperature": temperature},
        )

        raw_response, api_metadata = provider.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        trajectory.add_step(
            step_type="response",
            content=raw_response,
            metadata=api_metadata,
        )

        # Step 3: Parse response
        parsed_output = extract_response(task, raw_response)

        trajectory.add_step(
            step_type="parse",
            content=parsed_output,
            metadata={"raw_length": len(raw_response), "parsed_length": len(str(parsed_output))},
        )

        # Finalize trajectory
        trajectory.end_time = datetime.now().isoformat()
        trajectory.raw_output = raw_response
        # Store parsed output as-is (will be JSON encoded when saving the whole prediction dict)
        # Converting to string here would cause double encoding
        trajectory.final_prediction = parsed_output
        trajectory.success = True
        trajectory.token_usage = api_metadata.get("usage")
        trajectory.latency_ms = api_metadata.get("latency_ms")

    except Exception as e:
        trajectory.add_step(
            step_type="error",
            content=str(e),
            metadata={"traceback": traceback.format_exc()},
        )
        trajectory.end_time = datetime.now().isoformat()
        trajectory.success = False
        trajectory.error = str(e)
        logger.error(f"Error processing {instance_id}: {e}")

    return trajectory


def run_inference(
    task: str,
    model: str,
    dataset_path: str,
    output_dir: str = "results",
    prompt_style: str = "zero_shot",
    max_tokens: int = 1024,
    temperature: float = 0.2,
    run_id: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
    limit: Optional[int] = None,
    ollama_host: str = "http://localhost:11434",
    ground_truth_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run inference on an entire dataset.

    Args:
        task: Task name (adr, traceability, etc.)
        model: Model name (gpt-4, claude-3-opus, etc.)
        dataset_path: Path to dataset file
        output_dir: Directory to save results
        prompt_style: Prompt style (zero_shot, few_shot)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        run_id: Unique identifier for this run
        instance_ids: Optional list of specific instance IDs to process
        resume_from: Path to existing predictions file to resume from
        limit: Optional limit on number of instances to process (for testing)
        ground_truth_dir: Directory of ground truth diagram images (diagram task).
            When given, each prediction records its ground truth image so that
            evaluation and the judge can run without re-specifying the directory.

    Returns:
        Dictionary with summary statistics
    """
    start_time = time.time()

    # Generate run ID
    if run_id is None:
        run_id = f"{model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting inference run: {run_id}")
    logger.info(f"Task: {task}, Model: {model}, Prompt style: {prompt_style}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
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
            ground_truth_dir=ground_truth_dir,
        )
    else:
        # TODO: Add other task loaders
        dataset = load_dataset(task, dataset_path=dataset_path, instance_ids=instance_ids)

    # Apply limit if specified
    if limit is not None and limit > 0:
        original_count = len(dataset)
        dataset = dataset[:limit]
        logger.info(f"Limited to {len(dataset)} instances (from {original_count} total)")
    else:
        logger.info(f"Loaded {len(dataset)} instances")

    # Load existing predictions if resuming
    existing_predictions = {}
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from {resume_from}")
        with open(resume_from, "r") as f:
            for line in f:
                pred = json.loads(line)
                existing_predictions[pred[KEY_INSTANCE_ID]] = pred
        logger.info(f"Loaded {len(existing_predictions)} existing predictions")

    # Initialize provider and trajectory logger
    provider = get_provider(model, ollama_host=ollama_host)
    traj_logger = TrajectoryLogger(output_dir, run_id)

    # Setup output
    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_file = output_path / LOG_PREDICTIONS

    # Process instances
    predictions = []
    success_count = 0
    error_count = 0

    with open(predictions_file, "w") as f:
        for instance in tqdm(dataset, desc=f"Inference ({model})"):
            instance_id = instance[KEY_INSTANCE_ID]

            # Skip if already processed
            if instance_id in existing_predictions:
                predictions.append(existing_predictions[instance_id])
                f.write(json.dumps(existing_predictions[instance_id], ensure_ascii=False) + "\n")
                f.flush()
                continue

            # Run inference with trajectory
            trajectory = run_inference_single(
                task=task,
                instance=instance,
                provider=provider,
                prompt_style=prompt_style,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Debug: print response
            print(f"\n{'='*60}")
            print(f"Instance: {instance_id}")
            print(f"Success: {trajectory.success}")
            if trajectory.error:
                print(f"Error: {trajectory.error}")
            if trajectory.raw_output:
                print(f"Response preview: {trajectory.raw_output[:500]}...")
            print(f"{'='*60}\n")

            # Log trajectory
            traj_logger.log_trajectory(trajectory)

            # Create prediction record
            prediction = {
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: model,
                KEY_PREDICTION: trajectory.final_prediction if trajectory.final_prediction is not None else "",
                KEY_RAW_OUTPUT: trajectory.raw_output or "",
                "success": trajectory.success,
                "latency_ms": trajectory.latency_ms,
                "token_usage": trajectory.token_usage,
            }

            # For diagram, render the generated PlantUML to an image for evaluation
            # and carry the ground truth reference through to the prediction record
            if task == "diagram":
                from archbench.constants import KEY_GENERATED_IMAGE, KEY_GROUND_TRUTH_IMAGE
                if trajectory.success and trajectory.final_prediction:
                    image_path = render_plantuml(
                        puml_code=str(trajectory.final_prediction),
                        instance_id=instance_id,
                        output_dir=output_path / "generated_images",
                    )
                    prediction[KEY_GENERATED_IMAGE] = image_path or ""
                prediction[KEY_GROUND_TRUTH_IMAGE] = instance.get(KEY_GROUND_TRUTH_IMAGE) or ""

            predictions.append(prediction)

            # Write immediately (for resume capability)
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            f.flush()

            if trajectory.success:
                success_count += 1
            else:
                error_count += 1

    # Save all trajectories
    traj_logger.save_all()

    # Generate summary
    elapsed_time = time.time() - start_time
    summary = {
        "run_id": run_id,
        "task": task,
        "model": model,
        "prompt_style": prompt_style,
        "total_instances": len(dataset),
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / len(dataset) if dataset else 0,
        "elapsed_time_seconds": elapsed_time,
        "predictions_file": str(predictions_file),
        "trajectories_dir": str(traj_logger.output_dir),
    }

    # Save summary
    summary_file = output_path / "inference_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Inference complete: {success_count}/{len(dataset)} successful")
    logger.info(f"Results saved to {output_path}")

    return summary


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = ArgumentParser(
        description="Run ArchBench inference with trajectory logging",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        choices=list(TASKS.keys()),
        help="Task to run inference on",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4, claude-3-opus-20240229)",
    )
    parser.add_argument(
        "-d", "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset file",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Prompt style to use",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique identifier for this run",
    )
    parser.add_argument(
        "-i", "--instance_ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific instance IDs to process",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to existing predictions file to resume from",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (for ollama/ models)",
    )

    args = parser.parse_args()

    run_inference(
        task=args.task,
        model=args.model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        prompt_style=args.prompt_style,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        run_id=args.run_id,
        instance_ids=args.instance_ids,
        resume_from=args.resume_from,
        ollama_host=args.ollama_host,
    )


if __name__ == "__main__":
    main()
