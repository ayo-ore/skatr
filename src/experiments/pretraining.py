import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset

from src import models
from src.experiments.base_experiment import BaseExperiment

class PretrainingExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            if self.cfg.data.dedicated_test:
                return PretrainingDatasetByFile(self.cfg.data), PretrainingDatasetByFile(self.cfg.data, mode='test')
            else:
                return PretrainingDatasetByFile(self.cfg.data)
        else:
            if self.cfg.data.dedicated_test:
                return PretrainingDataset(self.cfg.data, self.device), PretrainingDataset(self.cfg.data, self.device, mode='test')
            else:
                return PretrainingDataset(self.cfg.data, self.device)

    def get_model(self):
        model_cls = getattr(models, self.cfg.model)
        return model_cls(self.cfg)
    
    def plot(self):
        raise NotImplementedError
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        raise NotImplementedError


class PretrainingDatasetByFile(Dataset):

    def __init__(self, cfg, mode='default'):
        if mode == 'test':
            test_dir = '/test'
        else: test_dir = ''
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}{test_dir}/run*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
        return X, 


class PretrainingDataset(Dataset):

    def __init__(self, cfg, device, mode='default'):
        if mode == 'test':
            test_dir = '/test'
        else: test_dir = ''
        self.files = sorted(glob(f'{cfg.dir}{test_dir}/run*.npz'))
        self.Xs = []
        
        for f in self.files:
            record = np.load(f)
            X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
            if cfg.on_gpu:
                X = X.to(device)
            self.Xs.append(X)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx],