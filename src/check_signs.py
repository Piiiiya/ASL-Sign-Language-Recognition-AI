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

print()
print(f"{'SIGN':<18}{'TRAIN':>8}{'VAL':>8}{'TEST':>8}")
print("-" * 42)

for sign in signs:
    train_count = (train["Gloss"] == sign).sum()
    val_count = (val["Gloss"] == sign).sum()
    test_count = (test["Gloss"] == sign).sum()

    print(
        f"{sign:<18}"
        f"{train_count:>8}"
        f"{val_count:>8}"
        f"{test_count:>8}"
    )