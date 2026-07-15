"""
Architecture View Generation evaluation metrics.

Implements image similarity metrics comparing a generated diagram image
against a ground truth diagram image:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- RMSE (Root Mean Squared Error)
- SAM  (Spectral Angle Mapper)
- SRE  (Signal to Reconstruction Error)
- UIQ  (Universal Image Quality index)

The generated image is resized to the ground truth dimensions before
comparison, mirroring the reference CodeToDiagram implementation.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# List of valid image extensions
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Metrics computed for the diagram task (order matters for reporting)
DIAGRAM_METRICS = ["ssim", "psnr", "rmse", "sam", "sre", "uiq"]


# =============================================================================
# Lazy Loading of Image Libraries (avoid torch/opencv import overhead)
# =============================================================================

_IMAGE_CACHE = {}


def _get_cv2():
    if "cv2" not in _IMAGE_CACHE:
        try:
            import cv2
            _IMAGE_CACHE["cv2"] = cv2
        except ImportError:
            logger.warning("opencv-python not installed. Install with: pip install archbench[diagram]")
            _IMAGE_CACHE["cv2"] = None
    return _IMAGE_CACHE["cv2"]


def _get_ssim():
    if "ssim" not in _IMAGE_CACHE:
        try:
            from skimage.metrics import structural_similarity
            _IMAGE_CACHE["ssim"] = structural_similarity
        except ImportError:
            logger.warning("scikit-image not installed. Install with: pip install archbench[diagram]")
            _IMAGE_CACHE["ssim"] = None
    return _IMAGE_CACHE["ssim"]


def _get_quality_metrics():
    if "quality" not in _IMAGE_CACHE:
        try:
            from image_similarity_measures.quality_metrics import psnr, rmse, sam, sre, uiq
            _IMAGE_CACHE["quality"] = {
                "psnr": psnr,
                "rmse": rmse,
                "sam": sam,
                "sre": sre,
                "uiq": uiq,
            }
        except ImportError:
            logger.warning(
                "image-similarity-measures not installed. Install with: pip install archbench[diagram]"
            )
            _IMAGE_CACHE["quality"] = None
    return _IMAGE_CACHE["quality"]


def _image_libs_available() -> bool:
    """Check whether the image similarity dependencies are installed."""
    return _get_cv2() is not None and _get_ssim() is not None and _get_quality_metrics() is not None


def _empty_metrics() -> Dict[str, float]:
    return {metric: 0.0 for metric in DIAGRAM_METRICS}


# =============================================================================
# Diagram Metrics
# =============================================================================

def compute_diagram_metrics(
    generated_image_path: str,
    ground_truth_image_path: str,
    compute_uiq: bool = True,
) -> Dict[str, float]:
    """
    Compute image similarity metrics for a single generated/ground-truth pair.

    Args:
        generated_image_path: Path to the generated diagram image
        ground_truth_image_path: Path to the reference diagram image
        compute_uiq: Whether to compute UIQ (slower; disabled skips it as None)

    Returns:
        Dictionary of metric scores (ssim, psnr, rmse, sam, sre, uiq)
    """
    if not _image_libs_available():
        logger.info("Image metrics skipped (image libraries not installed). "
                    "Install with: pip install archbench[diagram]")
        return {metric: None for metric in DIAGRAM_METRICS}

    cv2 = _get_cv2()
    structural_similarity = _get_ssim()
    quality = _get_quality_metrics()

    img_gt = cv2.imread(ground_truth_image_path)
    img_gen = cv2.imread(generated_image_path)

    if img_gt is None or img_gen is None:
        logger.warning(
            f"Could not load image(s): gt={ground_truth_image_path}, gen={generated_image_path}"
        )
        return _empty_metrics()

    # Resize generated image to ground truth dimensions before comparison
    img_gen_resized = cv2.resize(img_gen, (img_gt.shape[1], img_gt.shape[0]))

    metrics = {}
    try:
        metrics["ssim"] = float(
            structural_similarity(img_gt, img_gen_resized, channel_axis=-1, data_range=255)
        )
        metrics["psnr"] = float(quality["psnr"](img_gt, img_gen_resized))
        metrics["rmse"] = float(quality["rmse"](img_gt, img_gen_resized))
        metrics["sam"] = float(quality["sam"](img_gt, img_gen_resized))
        metrics["sre"] = float(quality["sre"](img_gt, img_gen_resized))
        metrics["uiq"] = float(quality["uiq"](img_gt, img_gen_resized)) if compute_uiq else None
    except Exception as e:
        logger.warning(f"Image metric computation failed: {e}")
        return _empty_metrics()

    return metrics


def compute_diagram_metrics_batch(
    generated_image_paths: List[str],
    ground_truth_image_paths: List[str],
    compute_uiq: bool = True,
) -> Dict[str, List[float]]:
    """
    Compute image similarity metrics for a batch of generated/ground-truth pairs.

    Args:
        generated_image_paths: List of generated diagram image paths
        ground_truth_image_paths: List of reference diagram image paths
        compute_uiq: Whether to compute UIQ

    Returns:
        Dictionary mapping metric names to lists of scores
    """
    n = len(generated_image_paths)
    assert len(ground_truth_image_paths) == n, "Generated and ground truth lists must have same length"

    results = {metric: [0.0] * n for metric in DIAGRAM_METRICS}

    for i, (gen_path, gt_path) in enumerate(zip(generated_image_paths, ground_truth_image_paths)):
        metrics = compute_diagram_metrics(gen_path, gt_path, compute_uiq=compute_uiq)
        for metric in DIAGRAM_METRICS:
            results[metric][i] = metrics.get(metric, 0.0)

    return results
