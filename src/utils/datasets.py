import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset

# from src.utils.augmentations import RotateAndReflect

class LCDataset(Dataset):

    def __init__(self, cfg, device, use_labels=True, preprocessing=None):

        self.use_labels = use_labels
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.Xs, self.ys = [], []

        dtype = torch.get_default_dtype()

        for f in self.files:
            record = np.load(f)

            # read and preprocess
            X = torch.from_numpy(record['image']).to(dtype) # TODO: Add option for `channels_last` memory format?
            for f in preprocessing['x']:
                X = f.forward(X)
            self.Xs.append(X)
            
            if use_labels:
                # read and preprocess
                y = torch.from_numpy(record['label']).to(dtype) # TODO: Cast with numpy before
                for f in preprocessing['y']:
                    y = f.forward(y)
                self.ys.append(y)
            
            if cfg.on_gpu:
                X = X.to(device)
                if use_labels:
                    y = y.to(device)
        
    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return (self.Xs[idx], self.ys[idx]) if self.use_labels else (self.Xs[idx],)
    

class LCDatasetByFile(Dataset):

    def __init__(self, cfg, use_labels=True, preprocessing=None):
        
        self.cfg = cfg
        self.use_labels = use_labels
        self.preprocessing = preprocessing

        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        
        # read
        X = torch.from_numpy(record['image']).to(self.dtype)
        # preprocess
        for f in self.preprocessing['x']:
            X = f.forward(X)
        
        if self.use_labels:
            # read
            y = torch.from_numpy(record['label']).to(self.dtype)
            # preprocess
            for f in self.preprocessing['y']:
                y = f.forward(y)

        return (X, y) if self.use_labels else (X,)


# class UnlabelledDataset(Dataset):

#     def __init__(self, cfg, device):
#         self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
#         self.Xs = []
        
#         for f in self.files:
#             record = np.load(f)
#             X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
#             if cfg.on_gpu:
#                 X = X.to(device)
#             self.Xs.append(X)


class SummarizedLCDataset(Dataset):

    def __init__(self, dataset, summary_net, device): #, augment=True):
        
        self.Xs = dataset.Xs
        self.ys = dataset.ys
        self.summary_net = summary_net

        # if augment:
        #     aug = RotateAndReflect()
        #     self.aug_Xs = []

        for i in range(len(self.Xs)):
            
            X = self.Xs[i]
            y = self.ys[i]
            
            X = X.to(device).unsqueeze(0)
            self.Xs[i] = self.summarize(X)
            
            # if augment: # this implementation leads to data leakage across training splits
            #     for x in aug.enumerate(X):
            #         self.Xs.append(self.summarize(x))
            #         self.ys.append(y)

    def summarize(self, x):
        with torch.no_grad():
            x = self.summary_net(x).squeeze(0)
            if not hasattr(self.summary_net, 'head'):
                x = x.mean(0) # (T, D) --> (D,)
        return x

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]