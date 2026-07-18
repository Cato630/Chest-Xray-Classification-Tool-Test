from pathlib import Path

import pandas as pd

from inference import format_prediction, predict_xray


TEST_CSV = Path("data/prepared/test.csv")


def main() -> None:
    test_dataframe = pd.read_csv(TEST_CSV)

    if test_dataframe.empty:
        raise ValueError("The test dataset is empty.")

    first_record = test_dataframe.iloc[0]

    image_path = Path(
        first_record["image_path"]
    )

    known_label = first_record["class_name"]

    prediction = predict_xray(image_path)

    print("Selected image")
    print("--------------")
    print(image_path.resolve())

    print("\nModel result")
    print("------------")
    print(format_prediction(prediction))

    print(
        f"\nKnown NIH binary label: {known_label}"
    )


if __name__ == "__main__":
    main()