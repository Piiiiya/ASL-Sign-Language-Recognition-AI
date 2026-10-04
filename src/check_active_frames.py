from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDMARK_DIR = PROJECT_DIR / "data" / "processed" / "landmarks"

SPLITS = ["train", "val", "test"]


def analyze_file(file_path):
    data = np.load(file_path)

    if data.ndim != 2 or data.shape[1] != 126:
        return None

    left = data[:, :63]
    right = data[:, 63:]

    left_missing = np.all(left == 0, axis=1)
    right_missing = np.all(right == 0, axis=1)

    # At least one hand detected
    active = ~(left_missing & right_missing)

    total = len(active)
    active_count = int(active.sum())

    return total, active_count


def main():
    print("=" * 80)
    print("ASL ACTIVE HAND-FRAME ANALYSIS")
    print("=" * 80)

    overall_total = 0
    overall_active = 0

    for split in SPLITS:
        split_dir = LANDMARK_DIR / split
        files = sorted(split_dir.glob("*.npy"))

        ratios = []
        active_counts = []

        for file_path in files:
            result = analyze_file(file_path)

            if result is None:
                continue

            total, active = result

            if total > 0:
                ratio = active / total * 100
                ratios.append(ratio)
                active_counts.append(active)

            overall_total += total
            overall_active += active

        ratios.sort()

        print(f"\n{'=' * 80}")
        print(f"{split.upper()}")
        print(f"{'=' * 80}")

        print(f"Videos analyzed       : {len(ratios)}")
        print(f"Minimum active frames : {min(active_counts)}")
        print(f"Maximum active frames : {max(active_counts)}")
        print(f"Average active frames : {sum(active_counts) / len(active_counts):.2f}")

        print(f"\nMinimum active %      : {min(ratios):.2f}%")
        print(f"Maximum active %      : {max(ratios):.2f}%")
        print(f"Average active %      : {sum(ratios) / len(ratios):.2f}%")
        print(f"Median active %       : {ratios[len(ratios) // 2]:.2f}%")

        # Count videos above useful thresholds
        for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            count = sum(r >= threshold for r in ratios)
            percentage = count / len(ratios) * 100

            print(
                f"Videos with >= {threshold:2d}% active frames : "
                f"{count:3d} ({percentage:.2f}%)"
            )

    print(f"\n{'=' * 80}")
    print("OVERALL")
    print(f"{'=' * 80}")

    print(f"Total frames          : {overall_total:,}")
    print(f"Active frames         : {overall_active:,}")

    active_percentage = (
        overall_active / overall_total * 100
        if overall_total
        else 0
    )

    print(f"Overall active %      : {active_percentage:.2f}%")
    print(
        f"Overall inactive %    : "
        f"{100 - active_percentage:.2f}%"
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()