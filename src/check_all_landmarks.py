from pathlib import Path
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

LANDMARK_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "landmarks"
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED = {
    "train": 383,
    "val": 79,
    "test": 298
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ASL CITIZEN - LANDMARK DATASET VERIFICATION")
    print("=" * 70)

    total_files = 0
    total_invalid = 0

    all_frame_counts = []

    # ========================================================
    # CHECK EACH SPLIT
    # ========================================================

    for split in ["train", "val", "test"]:

        split_dir = LANDMARK_DIR / split

        files = sorted(
            split_dir.glob("*.npy")
        )

        print()
        print("-" * 70)
        print(f"{split.upper()}")
        print("-" * 70)

        print(
            f"Expected files : {EXPECTED[split]}"
        )

        print(
            f"Found files    : {len(files)}"
        )

        if len(files) != EXPECTED[split]:

            print(
                "WARNING: File count does not match!"
            )

        valid = 0
        invalid = 0

        frame_counts = []

        for file_path in files:

            try:

                data = np.load(
                    file_path,
                    allow_pickle=False
                )

                # --------------------------------------------
                # Check dimensions
                # --------------------------------------------

                if data.ndim != 2:

                    print(
                        f"INVALID: {file_path.name} "
                        f"shape={data.shape}"
                    )

                    invalid += 1
                    continue

                # --------------------------------------------
                # Check feature count
                # --------------------------------------------

                if data.shape[1] != 126:

                    print(
                        f"INVALID: {file_path.name} "
                        f"shape={data.shape}"
                    )

                    invalid += 1
                    continue

                # --------------------------------------------
                # Check frame count
                # --------------------------------------------

                if data.shape[0] <= 0:

                    print(
                        f"INVALID: {file_path.name} "
                        f"has zero frames"
                    )

                    invalid += 1
                    continue

                # --------------------------------------------
                # Check NaN / Inf
                # --------------------------------------------

                if not np.isfinite(data).all():

                    print(
                        f"INVALID: {file_path.name} "
                        f"contains NaN/Inf"
                    )

                    invalid += 1
                    continue

                # --------------------------------------------
                # Valid
                # --------------------------------------------

                valid += 1

                frame_counts.append(
                    data.shape[0]
                )

                all_frame_counts.append(
                    data.shape[0]
                )

            except Exception as e:

                print(
                    f"INVALID: {file_path.name}"
                )

                print(
                    f"   Error: {repr(e)}"
                )

                invalid += 1

        total_files += len(files)
        total_invalid += invalid

        # ====================================================
        # SPLIT SUMMARY
        # ====================================================

        print()

        print(
            f"Valid files   : {valid}"
        )

        print(
            f"Invalid files : {invalid}"
        )

        if frame_counts:

            print(
                f"Min frames    : "
                f"{min(frame_counts)}"
            )

            print(
                f"Max frames    : "
                f"{max(frame_counts)}"
            )

            print(
                f"Average frames: "
                f"{np.mean(frame_counts):.2f}"
            )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    print(
        f"Total .npy files : {total_files}"
    )

    print(
        f"Invalid files    : {total_invalid}"
    )

    if all_frame_counts:

        print(
            f"Minimum frames   : "
            f"{min(all_frame_counts)}"
        )

        print(
            f"Maximum frames   : "
            f"{max(all_frame_counts)}"
        )

        print(
            f"Average frames   : "
            f"{np.mean(all_frame_counts):.2f}"
        )

        print(
            f"Median frames    : "
            f"{np.median(all_frame_counts):.2f}"
        )

    print()

    if (
        total_files == 760
        and total_invalid == 0
    ):

        print(
            "SUCCESS: ALL 760 LANDMARK FILES ARE VALID"
        )

        print(
            "SUCCESS: EVERY FILE HAS 126 FEATURES"
        )

    else:

        print(
            "WARNING: DATASET VERIFICATION FAILED"
        )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()