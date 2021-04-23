import csv
import os
import shutil
from pathlib import Path

import pandas as pd

DATA_ROOT = Path().home() / "ADNI_DATASET/ADNI1_Complete_1Yr_1.5T"
ADNI_DIR = DATA_ROOT / "ADNI"
SPLIT_DIR = "Classified/AD"
NII_OUT = DATA_ROOT / "Test2"
LABEL = "AD"


def split_subject_files() -> None:
    """This code is used to split the subject group folders into AD/CN/MCI folders"""

    whole_adni_df = pd.read_csv("ADNI1_Complete_1Yr_1.5T_9_22_2020.csv")
    groupby_subject = whole_adni_df.groupby("Subject").head(1)
    all_ad = groupby_subject[groupby_subject["Group"] == LABEL]
    all_subjects = all_ad["Subject"]

    for subject in all_subjects:
        os.chdir(str(ADNI_DIR))
        shutil.copytree(subject, SPLIT_DIR / subject)

# refered this code from https://stackoverflow.com/questions/59207743/using-os-walk-with-passing-a-variable/59208776#59208776
def copy_nii_info() -> None:
    """This code is used to take all the .nii files from subfolders recursively and copy them in new
    destination folder and also make entry in csv file for further uses"""

    with open("All_Data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label"])

        for root, _, files in os.walk(SPLIT_DIR):
            for file in files:
                print(file)
                writer.writerow([file, LABEL])
                path = os.path.join(root, file)
                shutil.copy2(path, NII_OUT)


if __name__ == "__main__":
    split_subject_files()
    copy_nii_info()
