"""
Architecture View Generation dataset loading.

Loads repository summaries (with architectural concern and behavior) that serve
as inference inputs, and resolves the ground truth diagram image used as the
evaluation reference.

Supported input formats:
- JSONL: one object per line with keys ``Repository Name``, ``summary``,
  ``Concern``, ``Behavior`` (and optionally ``repo_url``).
- CSV: the filtered dataset with the same column names.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from archbench.constants import (
    KEY_INSTANCE_ID,
    KEY_SUMMARY,
    KEY_CONCERN,
    KEY_BEHAVIOR,
    KEY_GROUND_TRUTH_IMAGE,
    DiagramInstance,
)
from archbench.tasks.diagram.grading import VALID_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "archbench" / "diagram"

# Column names as they appear in the CodeToDiagram dataset
COL_REPO_NAME = "Repository Name"
COL_SUMMARY = "summary"
COL_CONCERN = "Concern"
COL_BEHAVIOR = "Behavior"
COL_REPO_URL = "repo_url"


def clean_repo_name(repo_name: str) -> str:
    """
    Normalize a repository name into a filesystem-safe instance id.

    Mirrors the CodeToDiagram convention: forward/backward slashes become
    underscores and any trailing underscore is stripped.
    """
    return repo_name.replace("/", "_").replace("\\", "_").rstrip("_")


def _row_to_instance(row: Dict[str, Any]) -> Optional[DiagramInstance]:
    """Convert a raw dataset row into a diagram instance, or None if incomplete."""
    repo_name = row.get(COL_REPO_NAME) or row.get("Clean_Repo_Name")
    summary = row.get(COL_SUMMARY)
    if not repo_name or not summary:
        return None

    return {
        KEY_INSTANCE_ID: clean_repo_name(repo_name),
        KEY_SUMMARY: summary,
        KEY_CONCERN: row.get(COL_CONCERN, "general"),
        KEY_BEHAVIOR: row.get(COL_BEHAVIOR, "static"),
        COL_REPO_URL: row.get(COL_REPO_URL, ""),
        KEY_GROUND_TRUTH_IMAGE: None,  # Resolved later from ground_truth_dir
    }


def load_from_jsonl(jsonl_path: str) -> List[DiagramInstance]:
    """Load diagram instances from a JSONL file of repository summaries."""
    instances = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            instance = _row_to_instance(json.loads(line))
            if instance is not None:
                instances.append(instance)

    logger.info(f"Loaded {len(instances)} diagram instances from {jsonl_path}")
    return instances


def load_from_csv(csv_path: str) -> List[DiagramInstance]:
    """Load diagram instances from the filtered dataset CSV."""
    instances = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = _row_to_instance(row)
            if instance is not None:
                instances.append(instance)

    logger.info(f"Loaded {len(instances)} diagram instances from {csv_path}")
    return instances


def load_from_image_dir(ground_truth_dir: str) -> List[DiagramInstance]:
    """
    Build diagram instances directly from a directory of ground truth images.

    Each image becomes one instance whose id is the file stem and whose reference
    is the image path. Used for evaluation, where inference inputs (summaries) are
    not needed.
    """
    gt_path = Path(ground_truth_dir)
    instances = []
    for fname in sorted(gt_path.iterdir()):
        if fname.suffix.lower() in VALID_EXTENSIONS:
            instances.append({
                KEY_INSTANCE_ID: fname.stem,
                KEY_SUMMARY: "",
                KEY_CONCERN: "",
                KEY_BEHAVIOR: "",
                COL_REPO_URL: "",
                KEY_GROUND_TRUTH_IMAGE: str(fname),
            })

    logger.info(f"Loaded {len(instances)} ground truth images from {ground_truth_dir}")
    return instances


def resolve_ground_truth_images(
    instances: List[DiagramInstance],
    ground_truth_dir: str,
) -> List[DiagramInstance]:
    """
    Attach ground truth image paths to instances by matching on the instance id
    (stem of the image file), mirroring the folder-based comparison in the
    reference implementation.
    """
    gt_path = Path(ground_truth_dir)
    if not gt_path.is_dir():
        logger.warning(f"Ground truth directory not found: {ground_truth_dir}")
        return instances

    # Build stem -> path map for valid images
    stem_map = {}
    for fname in gt_path.iterdir():
        if fname.suffix.lower() in VALID_EXTENSIONS:
            stem_map[fname.stem] = str(fname)

    matched = 0
    for instance in instances:
        image_path = stem_map.get(instance[KEY_INSTANCE_ID])
        if image_path:
            instance[KEY_GROUND_TRUTH_IMAGE] = image_path
            matched += 1

    logger.info(f"Matched {matched}/{len(instances)} instances to ground truth images")
    return instances


def load_dataset(
    dataset_path: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
    ground_truth_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load the architecture view generation dataset.

    Args:
        dataset_path: Path to a JSONL/CSV file of repository summaries
        instance_ids: Optional filter for specific instances
        ground_truth_dir: Directory of ground truth diagram images (matched by stem)

    Returns:
        List of dataset instances
    """
    if dataset_path is None:
        raise ValueError(
            "Please provide a dataset_path (JSONL or CSV of repository summaries)."
        )

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # A directory of ground truth images: build one instance per image (stem = id).
    # This mirrors the folder-vs-folder comparison used during evaluation.
    if path.is_dir():
        dataset = load_from_image_dir(str(path))
        if instance_ids:
            instance_id_set = set(instance_ids)
            dataset = [d for d in dataset if d[KEY_INSTANCE_ID] in instance_id_set]
        return dataset

    if path.suffix == ".jsonl":
        dataset = load_from_jsonl(str(path))
    elif path.suffix == ".csv":
        dataset = load_from_csv(str(path))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            rows = list(data.values()) if isinstance(data, dict) else data
        dataset = [inst for inst in (_row_to_instance(r) for r in rows) if inst is not None]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .jsonl, .csv, or .json")

    # Resolve ground truth images if a directory is provided
    if ground_truth_dir:
        dataset = resolve_ground_truth_images(dataset, ground_truth_dir)

    # Filter by instance IDs if provided
    if instance_ids:
        instance_id_set = set(instance_ids)
        dataset = [d for d in dataset if d[KEY_INSTANCE_ID] in instance_id_set]

        found_ids = {d[KEY_INSTANCE_ID] for d in dataset}
        missing_ids = instance_id_set - found_ids
        if missing_ids:
            logger.warning(f"Missing {len(missing_ids)} instance IDs: {missing_ids}")

    return dataset


def extract_prediction(raw_output: str) -> str:
    """
    Extract PlantUML code from a raw model response.

    Strips markdown code fences (```plantuml / ```) and, when present, keeps only
    the ``@startuml ... @enduml`` block so the result is directly compilable.
    """
    if not raw_output:
        return ""

    text = raw_output.strip()

    # Remove surrounding markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop the opening fence (e.g. ```plantuml) and the closing fence
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Keep only the PlantUML block if the model added extra prose
    start = text.find("@startuml")
    end = text.rfind("@enduml")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + len("@enduml")]

    return text.strip()
