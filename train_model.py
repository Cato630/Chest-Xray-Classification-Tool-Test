from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "google/vit-base-patch16-224-in21k"

TRAIN_CSV = Path("data/prepared/train.csv")
VALIDATION_CSV = Path("data/prepared/validation.csv")
TEST_CSV = Path("data/prepared/test.csv")

MODEL_OUTPUT_DIRECTORY = Path("saved_model")
CHECKPOINT_DIRECTORY = Path("training_output")

RANDOM_SEED = 42
IMAGE_SIZE = 224

ID_TO_LABEL = {
    0: "Normal",
    1: "Abnormal",
}

LABEL_TO_ID = {
    "Normal": 0,
    "Abnormal": 1,
}

# Allows PIL to load some partially truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------
# Custom PyTorch dataset
# ---------------------------------------------------------

class ChestXrayDataset(Dataset):
    """
    PyTorch dataset for loading NIH chest X-ray images
    and their binary Normal/Abnormal labels.
    """

    def __init__(
        self,
        csv_path: Path,
        image_processor: AutoImageProcessor,
    ) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Prepared dataset was not found:\n"
                f"{csv_path.resolve()}"
            )

        self.dataframe = pd.read_csv(csv_path)
        self.image_processor = image_processor

        required_columns = {
            "image_path",
            "binary_label",
        }

        missing_columns = required_columns - set(
            self.dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{csv_path.name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.dataframe["binary_label"] = self.dataframe[
            "binary_label"
        ].astype(int)

        print(
            f"Loaded {csv_path.name}: "
            f"{len(self.dataframe):,} images"
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])
        label = int(row["binary_label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image could not be found:\n{image_path}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                processed_image = self.image_processor(
                    images=image,
                    return_tensors="pt",
                )

        except Exception as error:
            raise RuntimeError(
                f"Unable to process image:\n"
                f"{image_path}\n"
                f"Original error: {error}"
            ) from error

        pixel_values = processed_image[
            "pixel_values"
        ].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------------
# Batch data collator
# ---------------------------------------------------------

def collate_batch(
    examples: list[dict],
) -> dict[str, torch.Tensor]:
    """
    Combine individual image examples into one training batch.
    """

    pixel_values = torch.stack(
        [
            example["pixel_values"]
            for example in examples
        ]
    )

    labels = torch.stack(
        [
            example["labels"]
            for example in examples
        ]
    )

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

def compute_metrics(
    evaluation_prediction,
) -> dict[str, float]:
    """
    Calculate evaluation metrics from model predictions.
    """

    logits, true_labels = evaluation_prediction

    predicted_labels = np.argmax(
        logits,
        axis=1,
    )

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()[:, 1]

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="binary",
            zero_division=0,
        )
    )

    try:
        auc = roc_auc_score(
            true_labels,
            probabilities,
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "auc": float(auc),
    }


# ---------------------------------------------------------
# Display hardware information
# ---------------------------------------------------------

def display_hardware() -> None:
    print("Training hardware")
    print("-----------------")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "No CUDA GPU was detected. "
            "Training will use the CPU."
        )


# ---------------------------------------------------------
# Evaluate final test results
# ---------------------------------------------------------

def evaluate_test_set(
    trainer: Trainer,
    test_dataset: ChestXrayDataset,
) -> None:
    print("\nEvaluating final test dataset...")
    print("--------------------------------")

    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )

    for metric_name, metric_value in test_results.items():
        if isinstance(metric_value, float):
            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name}: "
                f"{metric_value}"
            )

    prediction_output = trainer.predict(
        test_dataset
    )

    predicted_labels = np.argmax(
        prediction_output.predictions,
        axis=1,
    )

    true_labels = prediction_output.label_ids

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    print("\nConfusion matrix")
    print("----------------")
    print(matrix)
    print(
        "\nMatrix format:"
        "\n[[True Normal, False Abnormal],"
        "\n [False Normal, True Abnormal]]"
    )


# ---------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    display_hardware()

    print("\nLoading image processor...")
    print("--------------------------")

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    train_dataset = ChestXrayDataset(
        TRAIN_CSV,
        image_processor,
    )

    validation_dataset = ChestXrayDataset(
        VALIDATION_CSV,
        image_processor,
    )

    test_dataset = ChestXrayDataset(
        TEST_CSV,
        image_processor,
    )

    print("\nLoading pretrained Vision Transformer...")
    print("----------------------------------------")

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIRECTORY),

        # Evaluate and save after each epoch.
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,

        # Training configuration.
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.10,

        # Model selection.
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Important for image training.
        remove_unused_columns=False,

        # Resource behavior.
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),

        # Keep only the two newest checkpoints.
        save_total_limit=2,

        # Avoid external experiment trackers.
        report_to="none",

        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_batch,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )

    print("\nBeginning Vision Transformer training...")
    print("----------------------------------------")

    trainer.train()

    print("\nTraining complete.")
    print("------------------")

    evaluate_test_set(
        trainer,
        test_dataset,
    )

    MODEL_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(MODEL_OUTPUT_DIRECTORY)
    )

    image_processor.save_pretrained(
        str(MODEL_OUTPUT_DIRECTORY)
    )

    print("\nSaved trained model")
    print("-------------------")
    print(MODEL_OUTPUT_DIRECTORY.resolve())


if __name__ == "__main__":
    main()