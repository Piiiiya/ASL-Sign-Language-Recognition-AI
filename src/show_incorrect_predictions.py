import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

file_path = PROJECT_DIR / "outputs" / "test_predictions_active.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("INCORRECT TEST PREDICTIONS")
print("=" * 80)

incorrect = [x for x in data if not x["correct"]]

print(f"\nTotal incorrect predictions: {len(incorrect)}\n")

for i, x in enumerate(incorrect, start=1):
    print(
        f"{i:2d}. "
        f"Actual: {x['actual_label']:15s} | "
        f"Predicted: {x['predicted_label']:15s} | "
        f"Confidence: {x['confidence']:.2f}%"
    )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)