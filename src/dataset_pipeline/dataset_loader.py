from pathlib import Path
import cv2
import numpy as np
from scipy.io import loadmat

class DatasetLoader:
    def __init__(self, image_dir, gt_dir):
        self.image_dir = Path(image_dir)
        self.gt_dir = Path(gt_dir)

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img

    def _load_annotation(self, img_path):
        # IMG_1.jpg → GT_IMG_1.mat
        gt_name = f"GT_{img_path.stem}.mat"
        gt_path = self.gt_dir / gt_name

        if not gt_path.exists():
            return None

        mat = loadmat(gt_path)

        # 🔥 THIS is the tricky part
        # Structure: mat["image_info"][0][0][0][0][0]
        points = mat["image_info"][0][0][0][0][0]

        return np.array(points)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = self._load_image(img_path)
        points = self._load_annotation(img_path)

        return {
            "image": image,
            "points": points,
            "count": 0 if points is None else len(points),
            "path": str(img_path)
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
