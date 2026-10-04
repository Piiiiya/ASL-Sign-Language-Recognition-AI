from pathlib import Path
import numpy as np
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent

LANDMARK_DIR = PROJECT_DIR / "data" / "processed" / "landmarks"

ZIP_ROOT = Path(r"C:\Users\hasibul shaikh\Downloads")

SPLITS = ["train", "val", "test"]

THRESHOLD = 10


def load_video_labels(split):
    """
    Load Video file -> Gloss mapping from the original ASL Citizen CSV.
    """

    csv_path = ZIP_ROOT / f"{split}.csv"

    # The CSVs are actually inside the ZIP, so this path normally
    # does not exist. This function is kept separate for clarity.
    return {}


def analyze_landmark(file_path):
    data = np.load(file_path)

    if data.ndim != 2 or data.shape[1] != 126:
        return None

    left = data[:, :63]
    right = data[:, 63:]

    left_missing = np.all(left == 0, axis=1)
    right_missing = np.all(right == 0, axis=1)

    active = ~(left_missing & right_missing)

    total = len(active)
    active_count = int(active.sum())

    percentage = (
        active_count / total * 100
        if total > 0
        else 0
    )

    return total, active_count, percentage


def main():

    print("=" * 80)
    print("LOW ACTIVE-FRAME VIDEO CHECK")
    print("=" * 80)

    for split in SPLITS:

        split_dir = LANDMARK_DIR / split

        results = []

        for file_path in split_dir.glob("*.npy"):

            result = analyze_landmark(file_path)

            if result is None:
                continue

            total, active, percentage = result

            results.append(
                (
                    percentage,
                    active,
                    total,
                    file_path.name
                )
            )

        results.sort()

        print(f"\n{'=' * 80}")
        print(f"{split.upper()}")
        print(f"{'=' * 80}")

        print(
            f"{'File':45} "
            f"{'Active':>8} "
            f"{'Total':>8} "
            f"{'Active %':>10}"
        )

        print("-" * 80)

        for percentage, active, total, filename in results[:15]:

            print(
                f"{filename:45} "
                f"{active:8d} "
                f"{total:8d} "
                f"{percentage:9.2f}%"
            )

        low_quality = [
            r for r in results
            if r[1] < THRESHOLD
        ]

        print("\nVideos with fewer than", THRESHOLD, "active frames:",
              len(low_quality))

    print("\n" + "=" * 80)
    print("CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()