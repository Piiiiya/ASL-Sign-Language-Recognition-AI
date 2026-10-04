import cv2
from pathlib import Path

print("Testing DirectShow webcam...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera opened successfully.")

for i in range(30):
    ret, frame = cap.read()

    if not ret:
        print(f"Frame {i}: FAILED")
        continue

    print(
        f"Frame {i}: shape={frame.shape}, "
        f"mean={frame.mean():.2f}, "
        f"min={frame.min()}, max={frame.max()}"
    )

    # Save frame 20
    if i == 20:
        output_path = Path("outputs") / "webcam_test.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), frame)
        print(f"\nSaved webcam image to: {output_path}")

cap.release()

print("Test finished.")