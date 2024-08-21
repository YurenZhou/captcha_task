import torch
import cv2
import numpy as np
import pickle
from pathlib import Path
import logging

from .config import CONFIG
from .utils import preprocess_image
from .captcha_cnn import CaptchaCNN
from .model_trainer import ModelTrainer

class Captcha:

    def __init__(self, load_model=True):
        if load_model:
            with open(CONFIG.LABEL_ENCODER_PATH, "rb") as file:
                self.label_encoder = pickle.load(file)
            with open(CONFIG.BEST_HYPER_PARAM_PATH, "rb") as file:
                best_hyper_params = pickle.load(file)

            self.model = CaptchaCNN(best_hyper_params)
            self.model.load_state_dict(torch.load(CONFIG.TRAINED_MODEL_PATH, weights_only=True))
        else:
            model_trainer = ModelTrainer()
            model_trainer.tune_hyper_params()
            model_trainer.train_final_model()
            model_trainer.save_model()

            self.model = model_trainer.final_model
            self.label_encoder = model_trainer.dataset.label_encoder

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """

        res_txt = self._infer_captcha(im_path)
        self._export_result(res_txt, save_path)

        logging.info(f"Result of infering image {im_path}: {res_txt}")

    def _infer_captcha(self, im_path):
        image = cv2.imread(im_path)
        segmented = preprocess_image(image)
        X = np.array(segmented)
        X = torch.Tensor(X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]))

        self.model.eval()
        with torch.no_grad():
            _, Y_hat = torch.max(self.model(X), 1)
        
        return "".join(self.label_encoder.inverse_transform(Y_hat))
    
    def _export_result(self, res_txt, save_path):
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(res_txt)
