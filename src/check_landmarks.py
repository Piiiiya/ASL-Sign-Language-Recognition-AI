import numpy as np
from pathlib import Path

file_path = Path(
    "data/processed/landmarks/train/014770041748456197-MOVIE.npy"
)

data = np.load(file_path)

print("File:", file_path)
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Minimum value:", data.min())
print("Maximum value:", data.max())
print("Non-zero values:", np.count_nonzero(data))

print()
print("=" * 50)
print("LANDMARK FILE CHECK COMPLETE")
print("=" * 50)

if data.shape[1] == 63:
    print("SUCCESS: 63 landmark features confirmed.")
else:
    print("ERROR: Unexpected feature count.")