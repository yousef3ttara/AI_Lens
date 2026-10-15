"""
AI Lens - Utility Functions & Preprocessing Pipeline
Handles model loading, image validation, MobileNetV2 preprocessing,
inference, uncertainty estimation, and diagnostic formatting.
"""

import json
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps

# Default paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "plant_disease_mobilenetv2.keras")
LEGACY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "plant_disease_mobilenetv2.h5")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "class_names.json")

# Target image dimensions required by MobileNetV2
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD_DEFAULT = 0.65


def load_class_names(path: str = CLASS_NAMES_PATH) -> List[str]:
    """Load canonical list of class names from JSON file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Class names file not found at: {path}")


def get_available_model_path() -> Optional[str]:
    """Check for standard saved model weights."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists(LEGACY_MODEL_PATH):
        return LEGACY_MODEL_PATH
    return None


def model_has_rescaling_layer(model: Any) -> bool:
    """Check if the model graph already contains an internal input Rescaling layer."""
    if model is not None and hasattr(model, "layers"):
        for layer in model.layers[:3]:  # Check initial layers
            layer_class = layer.__class__.__name__.lower()
            if "rescaling" in layer_class:
                return True
    return False


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    model: Optional[Any] = None,
) -> np.ndarray:
    """
    Preprocess an input PIL Image for MobileNetV2 inference:
    1. Standardize orientation from EXIF
    2. Convert to RGB (discarding alpha channels if present)
    3. Resize to 224x224 with high-quality resampling
    4. If the model already includes a Rescaling layer, preserve [0, 255] float range.
       Otherwise, normalize pixel values to [-1.0, 1.0].
    5. Expand dims to batch shape [1, 224, 224, 3]
    """
    # Fix orientation if EXIF metadata exists
    image = ImageOps.exif_transpose(image)

    # Ensure 3-channel RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to MobileNetV2 input resolution
    resized = image.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 numpy array [224, 224, 3]
    arr = np.array(resized, dtype=np.float32)

    # If the loaded model already has a Rescaling layer inside its graph,
    # pass raw [0, 255] pixels to avoid double-normalizing to near-zero constants
    if model is not None and model_has_rescaling_layer(model):
        final_arr = arr
    else:
        final_arr = (arr / 127.5) - 1.0

    # Expand batch dimension -> [1, 224, 224, 3]
    batched = np.expand_dims(final_arr, axis=0)
    return batched


def analyze_image_characteristics(image: Image.Image) -> Dict[str, Any]:
    """
    Inspect raw image characteristics to detect potentially invalid, blank,
    or non-plant inputs.
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    # Calculate channel means & standard deviations
    mean_r = np.mean(arr[:, :, 0])
    mean_g = np.mean(arr[:, :, 1])
    mean_b = np.mean(arr[:, :, 2])
    std_all = np.std(arr)

    # Low standard deviation indicates nearly solid/blank image
    is_blank = std_all < 12.0

    # Simple green-to-color ratio heuristic for foliage presence check
    total_mean = mean_r + mean_g + mean_b + 1e-6
    green_ratio = mean_g / total_mean

    return {
        "is_blank": is_blank,
        "std_dev": float(std_all),
        "green_ratio": float(green_ratio),
        "dimensions": image.size,
    }


def predict_leaf(
    model: Any,
    image: Image.Image,
    class_names: List[str],
    top_k: int = 3,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    """
    Execute inference on a leaf image using the loaded model.
    Returns predicted class, confidence, top-K distribution, and uncertainty flags.
    """
    # 1. Analyze image quality & characteristics
    img_stats = analyze_image_characteristics(image)

    # 2. Preprocess input tensor (respecting internal model rescaling if present)
    tensor = preprocess_image(image, model=model)

    # 3. Model forward pass
    preds = model.predict(tensor, verbose=0)[0]

    # Convert to standard python float array
    probs = np.array(preds, dtype=np.float64)

    # Sort descending
    top_indices = np.argsort(probs)[::-1][:top_k]

    top_class_idx = int(top_indices[0])
    top_class_name = class_names[top_class_idx]
    top_confidence = float(probs[top_class_idx])

    # Top-K summary
    top_predictions = []
    for idx in top_indices:
        c_name = class_names[idx]
        top_predictions.append({
            "class_name": c_name,
            "confidence": float(probs[idx]),
            "percentage": float(probs[idx] * 100.0),
        })

    # Evaluate uncertainty and potential out-of-distribution conditions
    margin = (
        float(probs[top_indices[0]] - probs[top_indices[1]])
        if len(top_indices) > 1
        else top_confidence
    )

    is_low_confidence = top_confidence < confidence_threshold
    is_ambiguous = margin < 0.15
    possible_non_plant = img_stats["is_blank"] or (
        top_confidence < 0.40 and img_stats["green_ratio"] < 0.28
    )

    return {
        "top_class": top_class_name,
        "confidence": top_confidence,
        "percentage": top_confidence * 100.0,
        "top_predictions": top_predictions,
        "is_low_confidence": is_low_confidence,
        "is_ambiguous": is_ambiguous,
        "possible_non_plant": possible_non_plant,
        "image_stats": img_stats,
        "confidence_threshold": confidence_threshold,
    }
