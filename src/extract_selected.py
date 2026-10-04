import zipfile
import pandas as pd
import os
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

zip_path = Path(os.environ["USERPROFILE"]) / "Downloads" / "ASL_Citizen.zip"

project_dir = Path(__file__).resolve().parent
output_dir = project_dir / "data" / "raw" / "ASL_Citizen"

# --------------------------------------------------
# Selected signs
# --------------------------------------------------

signs = [
    "APPLE",
    "DOG1",
    "BREAKFAST1",
    "DARK1",
    "DEAF1",
    "PARTY1",
    "DEVELOP1",
    "BEE1",
    "FINE1",
    "CHRISTMAS1",
    "HALLOWEEN1",
    "RIVER1",
    "LUNCH1",
    "HOSPITAL1",
    "MEAT1",
    "DINNER1",
    "EAT1",
    "MOVIE1",
    "SINK",
    "BASKETBALL1",
]

# --------------------------------------------------
# Create output folders
# --------------------------------------------------

for split in ["train", "val", "test"]:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

print("ZIP file:")
print(zip_path)

print()
print("Output folder:")
print(output_dir)

# --------------------------------------------------
# Open ZIP
# --------------------------------------------------

with zipfile.ZipFile(zip_path, "r") as z:

    # Read official split files
    train = pd.read_csv(
        z.open("ASL_Citizen/splits/train.csv")
    )

    val = pd.read_csv(
        z.open("ASL_Citizen/splits/val.csv")
    )

    test = pd.read_csv(
        z.open("ASL_Citizen/splits/test.csv")
    )

    # Extract selected videos from each split
    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    total = 0

    for split_name, df in splits.items():

        selected = df[df["Gloss"].isin(signs)]

        print()
        print(f"{split_name.upper()} videos: {len(selected)}")

        for _, row in selected.iterrows():

            video_name = row["Video file"]

            source_path = (
                "ASL_Citizen/videos/" + video_name
            )

            destination_dir = output_dir / split_name

            # Extract directly
            extracted_data = z.read(source_path)

            destination_file = destination_dir / video_name

            with open(destination_file, "wb") as f:
                f.write(extracted_data)

            total += 1

        print(f"{split_name.upper()} extraction complete.")

print()
print("=" * 50)
print("EXTRACTION COMPLETE")
print("=" * 50)
print(f"Total videos extracted: {total}")
print(f"Location: {output_dir}")