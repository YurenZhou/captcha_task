from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import SubsetRandomSampler, DataLoader
import optuna
from functools import partial
import logging
import pickle

from .captcha_dataset import CaptchaDataset
from .config import CONFIG
from .captcha_cnn import CaptchaCNN

class ModelTrainer:

    def __init__(self):
        self.dataset = CaptchaDataset(CONFIG.IMAGE_PATH, CONFIG.LABEL_PATH, transform=transforms.ToTensor())
        self.best_hyper_params = None
        self.final_model = None

    def tune_hyper_params(self):
        logging.info("Tuning hyper-parameters with cross-validation ...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self._optuna_objective, n_trials=CONFIG.HYPER_TUNING_TRIALS)
        self.best_hyper_params = study.best_params

    def train_final_model(self):
        logging.info("Training the final model using the best hyper-parameters ...")
        self.final_model = CaptchaCNN(self.best_hyper_params)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.final_model.parameters(), lr=self.best_hyper_params['learning_rate'])
        train_loader = DataLoader(self.dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
        for epoch in range(CONFIG.NUM_EPOCH):
            self.final_model.train()
            running_loss = 0.0
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                outputs = self.final_model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            logging.info(f'Epoch [{epoch + 1}/NUM_EPOCH], Loss: {running_loss / len(train_loader):.4f}')
        
    def save_model(self):
        with open(CONFIG.LABEL_ENCODER_PATH, 'wb') as file:
            pickle.dump(self.dataset.label_encoder, file)
        with open(CONFIG.BEST_HYPER_PARAM_PATH, 'wb') as file:
            pickle.dump(self.best_hyper_params, file)
        torch.save(self.final_model.state_dict(), CONFIG.TRAINED_MODEL_PATH)

    def _optuna_objective(self, trial):
        # Parameters to be tuned
        learning_rate = trial.suggest_float("learning_rate", CONFIG.LEARNING_RATE_LOWER, CONFIG.LEARNING_RATE_UPPER)
        hyper_params = {}
        hyper_params["num_filters"] = trial.suggest_int("num_filters", CONFIG.NUM_FILTERS_LOWER, CONFIG.NUM_FILTERS_UPPER)
        hyper_params["dropout_rate"] = trial.suggest_float("dropout_rate", CONFIG.DROPOUT_RATE_LOWER, CONFIG.DROPOUT_RATE_UPPER)
        
        # Leave-One-Out Cross Validation
        correct, total = 0, 0
        loo = LeaveOneOut()
        for train_idx, val_idx in loo.split(self.dataset):
            
            train_subsampler = SubsetRandomSampler(train_idx)
            train_loader = DataLoader(self.dataset, batch_size=CONFIG.BATCH_SIZE, sampler=train_subsampler)
            
            model = CaptchaCNN(hyper_params)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            for _ in range(CONFIG.NUM_EPOCH):
                model.train()
                for batch_X, batch_Y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_Y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                val_x, val_y = self.dataset[val_idx[0]]
                outputs = model(val_x)
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += int(predicted == val_y)
        
        return correct / total