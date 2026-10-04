from pathlib import Path
import numpy as np
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDMARK_DIR = PROJECT_DIR / "data" / "processed" / "landmarks"

SPLITS = ["train", "val", "test"]


def analyze_file(file_path):
    data = np.load(file_path)

    # Expected shape: (frames, 126)
    if data.ndim != 2 or data.shape[1] != 126:
        return None

    # Split into left and right hand
    left = data[:, :63]
    right = data[:, 63:]

    # A hand is considered missing when all 63 values are zero
    left_missing = np.all(left == 0, axis=1)
    right_missing = np.all(right == 0, axis=1)

    both_missing = left_missing & right_missing
    at_least_one_hand = ~both_missing
    both_hands = ~left_missing & ~right_missing

    total_frames = data.shape[0]

    return {
        "frames": total_frames,
        "left_missing": int(left_missing.sum()),
        "right_missing": int(right_missing.sum()),
        "both_missing": int(both_missing.sum()),
        "at_least_one_hand": int(at_least_one_hand.sum()),
        "both_hands": int(both_hands.sum()),
    }


def percent(value, total):
    if total == 0:
        return 0.0
    return value / total * 100


def main():
    print("=" * 75)
    print("ASL LANDMARK QUALITY CHECK")
    print("=" * 75)

    overall = {
        "files": 0,
        "frames": 0,
        "left_missing": 0,
        "right_missing": 0,
        "both_missing": 0,
        "at_least_one_hand": 0,
        "both_hands": 0,
    }

    for split in SPLITS:
        split_dir = LANDMARK_DIR / split
        files = sorted(split_dir.glob("*.npy"))

        print(f"\n{'=' * 75}")
        print(f"{split.upper()} SPLIT")
        print(f"{'=' * 75}")

        split_stats = {
            "files": 0,
            "frames": 0,
            "left_missing": 0,
            "right_missing": 0,
            "both_missing": 0,
            "at_least_one_hand": 0,
            "both_hands": 0,
        }

        for file_path in files:
            result = analyze_file(file_path)

            if result is None:
                print(f"INVALID: {file_path.name}")
                continue

            split_stats["files"] += 1

            for key in result:
                split_stats[key] += result[key]

        print(f"Videos/files              : {split_stats['files']}")
        print(f"Total frames              : {split_stats['frames']:,}")

        print(
            f"Left hand missing         : "
            f"{split_stats['left_missing']:,} "
            f"({percent(split_stats['left_missing'], split_stats['frames']):.2f}%)"
        )

        print(
            f"Right hand missing        : "
            f"{split_stats['right_missing']:,} "
            f"({percent(split_stats['right_missing'], split_stats['frames']):.2f}%)"
        )

        print(
            f"Both hands missing        : "
            f"{split_stats['both_missing']:,} "
            f"({percent(split_stats['both_missing'], split_stats['frames']):.2f}%)"
        )

        print(
            f"At least one hand         : "
            f"{split_stats['at_least_one_hand']:,} "
            f"({percent(split_stats['at_least_one_hand'], split_stats['frames']):.2f}%)"
        )

        print(
            f"Both hands detected       : "
            f"{split_stats['both_hands']:,} "
            f"({percent(split_stats['both_hands'], split_stats['frames']):.2f}%)"
        )

        # Add to overall
        for key in split_stats:
            overall[key] += split_stats[key]

    print(f"\n{'=' * 75}")
    print("OVERALL")
    print(f"{'=' * 75}")

    print(f"Videos/files              : {overall['files']}")
    print(f"Total frames              : {overall['frames']:,}")

    print(
        f"Left hand missing         : "
        f"{overall['left_missing']:,} "
        f"({percent(overall['left_missing'], overall['frames']):.2f}%)"
    )

    print(
        f"Right hand missing        : "
        f"{overall['right_missing']:,} "
        f"({percent(overall['right_missing'], overall['frames']):.2f}%)"
    )

    print(
        f"Both hands missing        : "
        f"{overall['both_missing']:,} "
        f"({percent(overall['both_missing'], overall['frames']):.2f}%)"
    )

    print(
        f"At least one hand         : "
        f"{overall['at_least_one_hand']:,} "
        f"({percent(overall['at_least_one_hand'], overall['frames']):.2f}%)"
    )

    print(
        f"Both hands detected       : "
        f"{overall['both_hands']:,} "
        f"({percent(overall['both_hands'], overall['frames']):.2f}%)"
    )

    print("\n" + "=" * 75)
    print("INTERPRETATION")
    print("=" * 75)

    missing_pct = percent(
        overall["both_missing"],
        overall["frames"]
    )

    if missing_pct < 5:
        print("GOOD: Very few frames have both hands missing.")
    elif missing_pct < 15:
        print("WARNING: Some frames have both hands missing.")
    else:
        print("PROBLEM: Many frames have both hands missing.")

    print("\nQuality check completed.")


if __name__ == "__main__":
    main()