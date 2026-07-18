#project1 model traininG


from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import transformers
import datasets
import sklearn
import pandas as pd
import numpy as np
import gradio as gr



# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

DATA_DIRECTORY = Path("data")

METADATA_PATH = DATA_DIRECTORY / "Data_Entry_2017_v2020.csv"

IMAGE_DIRECTORY = DATA_DIRECTORY / "images"

OUTPUT_DIRECTORY = DATA_DIRECTORY / "prepared"

RANDOM_SEED = 42

NORMAL_SAMPLE_SIZE = 1000
ABNORMAL_SAMPLE_SIZE = 1000


# ---------------------------------------------------------
# Load NIH metadata
# ---------------------------------------------------------

def load_metadata() -> pd.DataFrame:
    """
    Load and validate the NIH ChestX-ray14 metadata file.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "The NIH metadata file could not be found.\n"
            f"Expected location: {METADATA_PATH.resolve()}"
        )

    dataframe = pd.read_csv(METADATA_PATH)

    required_columns = {
        "Image Index",
        "Finding Labels",
        "Patient ID",
        "Patient Age",
        "Patient Sex",
        "View Position",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The metadata file is missing these required columns:\n"
            f"{sorted(missing_columns)}"
        )

    print("Metadata loaded successfully.")
    print(f"Total metadata rows: {len(dataframe):,}")

    return dataframe


# ---------------------------------------------------------
# Locate all downloaded images
# ---------------------------------------------------------

def build_image_lookup() -> dict[str, Path]:
    """
    Recursively search the image directory and create a lookup
    connecting each filename to its complete file path.
    """

    if not IMAGE_DIRECTORY.exists():
        raise FileNotFoundError(
            "The image directory could not be found.\n"
            f"Expected location: {IMAGE_DIRECTORY.resolve()}"
        )

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
    }

    image_lookup = {}

    for image_path in IMAGE_DIRECTORY.rglob("*"):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in image_extensions
        ):
            image_lookup[image_path.name] = image_path

    print(f"Downloaded image files found: {len(image_lookup):,}")

    if not image_lookup:
        raise ValueError(
            "No image files were found inside the image directory."
        )

    return image_lookup


# ---------------------------------------------------------
# Match metadata rows to images
# ---------------------------------------------------------

def match_images_to_metadata(
    dataframe: pd.DataFrame,
    image_lookup: dict[str, Path],
) -> pd.DataFrame:
    """
    Match every metadata row to the downloaded image file.
    """

    dataframe = dataframe.copy()

    dataframe["image_path"] = dataframe[
        "Image Index"
    ].map(image_lookup)

    dataframe["image_exists"] = dataframe[
        "image_path"
    ].notna()

    images_found = int(dataframe["image_exists"].sum())
    images_missing = len(dataframe) - images_found

    print("\nImage matching results")
    print("----------------------")
    print(f"Metadata images found: {images_found:,}")
    print(f"Metadata images missing: {images_missing:,}")

    dataframe = dataframe[
        dataframe["image_exists"]
    ].copy()

    dataframe["image_path"] = dataframe[
        "image_path"
    ].astype(str)

    return dataframe


# ---------------------------------------------------------
# Create binary labels
# ---------------------------------------------------------

def create_binary_labels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert NIH Finding Labels into binary labels.

    0 = Normal
    1 = Abnormal
    """

    dataframe = dataframe.copy()

    dataframe["binary_label"] = (
        dataframe["Finding Labels"] != "No Finding"
    ).astype(int)

    dataframe["class_name"] = dataframe[
        "binary_label"
    ].map(
        {
            0: "Normal",
            1: "Abnormal",
        }
    )

    print("\nAvailable class counts")
    print("----------------------")
    print(
        dataframe["class_name"]
        .value_counts()
        .to_string()
    )

    return dataframe


# ---------------------------------------------------------
# Create a balanced prototype dataset
# ---------------------------------------------------------

def create_balanced_sample(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select equal numbers of normal and abnormal images.
    """

    normal_dataframe = dataframe[
        dataframe["binary_label"] == 0
    ]

    abnormal_dataframe = dataframe[
        dataframe["binary_label"] == 1
    ]

    normal_count = min(
        NORMAL_SAMPLE_SIZE,
        len(normal_dataframe),
    )

    abnormal_count = min(
        ABNORMAL_SAMPLE_SIZE,
        len(abnormal_dataframe),
    )

    balanced_count = min(
        normal_count,
        abnormal_count,
    )

    if balanced_count == 0:
        raise ValueError(
            "A balanced dataset could not be created. "
            "At least one normal and one abnormal image are required."
        )

    sampled_normal = normal_dataframe.sample(
        n=balanced_count,
        random_state=RANDOM_SEED,
    )

    sampled_abnormal = abnormal_dataframe.sample(
        n=balanced_count,
        random_state=RANDOM_SEED,
    )

    balanced_dataframe = pd.concat(
        [
            sampled_normal,
            sampled_abnormal,
        ],
        ignore_index=True,
    )

    balanced_dataframe = balanced_dataframe.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print("\nBalanced prototype dataset")
    print("--------------------------")
    print(
        balanced_dataframe["class_name"]
        .value_counts()
        .to_string()
    )

    print(
        f"Total selected images: "
        f"{len(balanced_dataframe):,}"
    )

    return balanced_dataframe


# ---------------------------------------------------------
# Split by patient
# ---------------------------------------------------------

def split_by_patient(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split the dataset by Patient ID.

    This helps prevent images from the same patient from appearing
    in both training and evaluation data.
    """

    patient_labels = (
        dataframe.groupby("Patient ID")["binary_label"]
        .max()
        .reset_index()
    )

    train_patients, remaining_patients = train_test_split(
        patient_labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=patient_labels["binary_label"],
    )

    validation_patients, test_patients = train_test_split(
        remaining_patients,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=remaining_patients["binary_label"],
    )

    train_patient_ids = set(
        train_patients["Patient ID"]
    )

    validation_patient_ids = set(
        validation_patients["Patient ID"]
    )

    test_patient_ids = set(
        test_patients["Patient ID"]
    )

    train_dataframe = dataframe[
        dataframe["Patient ID"].isin(train_patient_ids)
    ].copy()

    validation_dataframe = dataframe[
        dataframe["Patient ID"].isin(
            validation_patient_ids
        )
    ].copy()

    test_dataframe = dataframe[
        dataframe["Patient ID"].isin(test_patient_ids)
    ].copy()

    print("\nPatient-level data split")
    print("------------------------")
    print(f"Training images: {len(train_dataframe):,}")
    print(
        f"Validation images: "
        f"{len(validation_dataframe):,}"
    )
    print(f"Testing images: {len(test_dataframe):,}")

    print(
        f"\nTraining patients: "
        f"{train_dataframe['Patient ID'].nunique():,}"
    )

    print(
        f"Validation patients: "
        f"{validation_dataframe['Patient ID'].nunique():,}"
    )

    print(
        f"Testing patients: "
        f"{test_dataframe['Patient ID'].nunique():,}"
    )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


# ---------------------------------------------------------
# Save prepared CSV files
# ---------------------------------------------------------

def save_prepared_data(
    full_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """
    Save the prepared metadata splits as CSV files.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns_to_save = [
        "Image Index",
        "image_path",
        "Finding Labels",
        "binary_label",
        "class_name",
        "Patient ID",
        "Patient Age",
        "Patient Sex",
        "View Position",
    ]

    full_dataframe[columns_to_save].to_csv(
        OUTPUT_DIRECTORY / "balanced_dataset.csv",
        index=False,
    )

    train_dataframe[columns_to_save].to_csv(
        OUTPUT_DIRECTORY / "train.csv",
        index=False,
    )

    validation_dataframe[columns_to_save].to_csv(
        OUTPUT_DIRECTORY / "validation.csv",
        index=False,
    )

    test_dataframe[columns_to_save].to_csv(
        OUTPUT_DIRECTORY / "test.csv",
        index=False,
    )

    print("\nPrepared files saved")
    print("--------------------")

    for filename in [
        "balanced_dataset.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
    ]:
        output_path = OUTPUT_DIRECTORY / filename
        print(output_path.resolve())


# ---------------------------------------------------------
# Main Phase 4 workflow
# ---------------------------------------------------------

def main() -> None:
    metadata = load_metadata()

    image_lookup = build_image_lookup()

    matched_data = match_images_to_metadata(
        metadata,
        image_lookup,
    )

    labeled_data = create_binary_labels(
        matched_data
    )

    balanced_data = create_balanced_sample(
        labeled_data
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_by_patient(
        balanced_data
    )

    save_prepared_data(
        balanced_data,
        train_data,
        validation_data,
        test_data,
    )

    print(
        "\nPhase 4 dataset preparation completed successfully."
    )


if __name__ == "__main__":
    main()