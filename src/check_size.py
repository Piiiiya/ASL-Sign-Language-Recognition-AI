import zipfile
import pandas as pd
import os

zip_path = os.path.expandvars(
    r"$USERPROFILE\Downloads\ASL_Citizen.zip"
)

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

with zipfile.ZipFile(zip_path) as z:
    train = pd.read_csv(
        z.open("ASL_Citizen/splits/train.csv")
    )
    val = pd.read_csv(
        z.open("ASL_Citizen/splits/val.csv")
    )
    test = pd.read_csv(
        z.open("ASL_Citizen/splits/test.csv")
    )

    all_data = pd.concat([train, val, test])

    selected = all_data[all_data["Gloss"].isin(signs)]

    zip_files = {
        name: info.file_size
        for name, info in (
            (info.filename, info)
            for info in z.infolist()
        )
    }

    total_size = 0
    found = 0
    missing = []

    for video in selected["Video file"]:
        path = "ASL_Citizen/videos/" + video

        if path in zip_files:
            total_size += zip_files[path]
            found += 1
        else:
            missing.append(video)

print()
print("Selected videos:", len(selected))
print("Videos found in ZIP:", found)
print("Videos missing:", len(missing))
print()
print(
    "Uncompressed video size: "
    f"{total_size / (1024 ** 3):.2f} GB"
)
print(
    "Uncompressed video size: "
    f"{total_size / (1024 ** 2):.2f} MB"
)