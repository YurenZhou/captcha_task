from torch.utils.data import Dataset
from pathlib import Path
import re
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

from .utils import preprocess_image


class CaptchaDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.X, self.Y, self.label_encoder = self._load_data_from_files(image_path, label_path)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        if self.transform:
            X = self.transform(X)

        return X, Y
    
    def _extract_file_ids(self, files):
        id_to_file = {}
        for file in files:
            id = re.search("\d+\Z", file.stem).group()
            id_to_file[id] = file
        return id_to_file

    def _load_data_from_files(self, image_path, label_path):

        image_files = self._extract_file_ids(sorted(list(Path(image_path).rglob("*.jpg"))))
        label_files = self._extract_file_ids(sorted(list(Path(label_path).rglob("*.txt"))))
        if len(image_files) != len(label_files):
            logging.warning("The number of images and labels does not match!")
        
        X = []
        Y = []
        for id, img_file in image_files.items():
            if id not in label_files:
                logging.warning(f"Image {img_file} does not have the corresponding label!")
                continue
            image = cv2.imread(img_file)
            segmented = preprocess_image(image)
            label = Path(label_files[id]).read_text().strip()
            if len(segmented) != len(label):
                raise Exception(f"ERROR: Image and label does not match for file {img_file}!")
            for char_img, char_lbl in zip(segmented, list(label)):
                X.append(char_img)
                Y.append(char_lbl)
        
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        Y = np.array(Y)
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        
        return X, Y, label_encoder