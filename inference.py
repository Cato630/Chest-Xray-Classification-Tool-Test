from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_DIRECTORY = Path("saved_model")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load model and processor once
# ---------------------------------------------------------

def load_model():
    """
    Load the trained image processor and Vision Transformer.

    The model is moved to the GPU when CUDA is available.
    """

    if not MODEL_DIRECTORY.exists():
        raise FileNotFoundError(
            "The saved_model directory was not found.\n"
            f"Expected location: {MODEL_DIRECTORY.resolve()}"
        )

    required_files = [
        MODEL_DIRECTORY / "config.json",
        MODEL_DIRECTORY / "preprocessor_config.json",
    ]

    missing_files = [
        str(file_path)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The saved model is incomplete. Missing files:\n"
            + "\n".join(missing_files)
        )

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_DIRECTORY
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIRECTORY
    )

    model.to(DEVICE)
    model.eval()

    return image_processor, model


IMAGE_PROCESSOR, MODEL = load_model()


# ---------------------------------------------------------
# Prepare an input image
# ---------------------------------------------------------

def prepare_image(
    image_input: str | Path | Image.Image,
) -> Image.Image:
    """
    Convert a file path or PIL image into an RGB PIL image.
    """

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    image_path = Path(image_input)

    if not image_path.exists():
        raise FileNotFoundError(
            f"The image was not found:\n{image_path.resolve()}"
        )

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    except Exception as error:
        raise ValueError(
            f"The file could not be opened as an image:\n"
            f"{image_path.resolve()}"
        ) from error


# ---------------------------------------------------------
# Run model inference
# ---------------------------------------------------------

def predict_xray(
    image_input: str | Path | Image.Image,
) -> dict[str, Any]:
    """
    Analyze one chest X-ray image.

    Returns:
        predicted_label
        confidence
        probabilities
        device
    """

    image = prepare_image(image_input)

    inputs = IMAGE_PROCESSOR(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in inputs.items()
    }

    with torch.inference_mode():
        outputs = MODEL(**inputs)

        probabilities_tensor = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

    probabilities = {}

    for class_id, probability in enumerate(
        probabilities_tensor
    ):
        label = MODEL.config.id2label.get(
            class_id,
            str(class_id),
        )

        probabilities[label] = float(
            probability.detach().cpu().item()
        )

    probabilities = dict(
        sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    predicted_label = next(iter(probabilities))

    confidence = probabilities[predicted_label]

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "device": str(DEVICE),
    }


# ---------------------------------------------------------
# Format results for display
# ---------------------------------------------------------

def format_prediction(
    prediction: dict[str, Any],
) -> str:
    """
    Convert prediction results into readable text.
    """

    predicted_label = prediction["predicted_label"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    lines = [
        f"Prediction: {predicted_label}",
        f"Confidence: {confidence * 100:.2f}%",
        "",
        "Class probabilities:",
    ]

    for label, probability in probabilities.items():
        lines.append(
            f"- {label}: {probability * 100:.2f}%"
        )

    lines.extend(
        [
            "",
            "Educational-use notice:",
            (
                "This prediction is generated by an experimental "
                "model and is not a clinical diagnosis."
            ),
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# Optional direct test
# ---------------------------------------------------------

def main() -> None:
    print("Inference module loaded successfully.")
    print(f"Selected device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        print(
            f"PyTorch CUDA version: {torch.version.cuda}"
        )


if __name__ == "__main__":
    main()