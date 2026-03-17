import gdown
import zipfile
import shutil
import os

FILE_ID = "1g-FZzbF0DrDnuMsVXcWh5t3xKAJ9b-N1"
ZIP_FILE = "dataset.zip"
DATASET_DIR = "dataset"

url = f"https://drive.google.com/uc?id={FILE_ID}"

print("Pulling dataset from Google Drive...")

gdown.download(url, ZIP_FILE, quiet=False)

if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(DATASET_DIR)

os.remove(ZIP_FILE)

print("Dataset updated locally.")