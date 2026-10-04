from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "confusion_samples"
    / "BREAKFAST1"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "BREAKFAST1_contact_sheet.jpg"
)

images = []

for video_dir in sorted(INPUT_DIR.glob("video_*")):
    for image_path in sorted(video_dir.glob("frame_*.jpg")):
        img = Image.open(image_path).convert("RGB")
        images.append((video_dir.name, image_path.name, img))

if not images:
    raise FileNotFoundError(
        f"No images found in: {INPUT_DIR}"
    )

# Thumbnail size
thumb_w = 320
thumb_h = 240

# Grid: 3 columns × 5 rows
cols = 3
rows = 5

label_h = 35

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * (thumb_h + label_h)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (video_name, frame_name, img) in enumerate(images):

    img.thumbnail((thumb_w - 10, thumb_h - 10))

    x = (i % cols) * thumb_w
    y = (i // cols) * (thumb_h + label_h)

    # Center image
    img_x = x + (thumb_w - img.width) // 2
    img_y = y + (thumb_h - img.height) // 2

    sheet.paste(img, (img_x, img_y))

    draw.text(
        (x + 5, y + thumb_h),
        f"{video_name} / {frame_name}",
        fill="black"
    )

sheet.save(
    OUTPUT_FILE,
    quality=95
)

print("=" * 70)
print("CONTACT SHEET CREATED")
print("=" * 70)
print(f"Images: {len(images)}")
print(f"Saved: {OUTPUT_FILE}")