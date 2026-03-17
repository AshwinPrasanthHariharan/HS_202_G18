import shutil
import hashlib

print("Creating dataset zip...")

shutil.make_archive("dataset", "zip", "dataset")

def sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

hash_value = sha256("dataset.zip")

with open("dataset.sha256", "w") as f:
    f.write(hash_value)

print("Checksum updated:", hash_value)
print("Upload dataset.zip to Google Drive and replace old file.")
print("Link: https://drive.google.com/drive/folders/1CjpIhF7LBYsFMFngy_TTkUxclgdYSyXX?usp=drive_link")