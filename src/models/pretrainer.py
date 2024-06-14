import torch
import torch.nn as nn
import random
from omegaconf import DictConfig

from src import networks
from src.models.base_model import Model

class Pretrainer(Model):

    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.predictor = networks.MLP(cfg.predictor)
        self.student = self.net
        self.teacher = self.net.__class__(cfg.net)

        ''' 
        match cfg.sim:
            case 'l1': self.sim = nn.L1Loss(reduction='mean')
            case 'l2': self.sim = nn.MSELoss(reduction='mean')
            case 'cosine': self.sim = -nn.CosineSimilarity(dim=1, eps=1e-6)
        Ayo uses different functions:
        '''
        match cfg.sim:
            case 'l1': self.sim = lambda x1, x2: (x1-x2).abs().mean(1)
            case 'l2': self.sim = lambda x1, x2: nn.functional.mse_loss(x1, x2)
            case 'cosine': self.sim = -nn.CosineSimilarity(dim=1, eps=1e-6)
        self.norm = nn.BatchNorm1d(cfg.latent_dim)

    def batch_loss(self, batch):        

        # augment
        x1 = augment(batch[0], include_identity=True) if self.cfg.augment else x1
        x2 = augment(x1) if self.cfg.augment else x1

        # embed masked batch
        embedding = self.student(x1, mask=self.cfg.mask)

        # embed full batch without grads
        with torch.no_grad():
            target = self.teacher(x2)
            if self.cfg.norm_target:
                target = self.norm(target)

        # predict teacher embedding from student embedding
        pred = self.predictor(embedding)

        # similarity loss
        loss = self.sim(pred, target)
        
        return loss.mean()
    
    def update(self, optimizer, loss):
        
        # student update
        super().update(optimizer, loss)

        # teacher update via exponential moving average of student
        tau = self.cfg.ema_momentum
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt = tau*pt + (1-tau)*ps

    def forward(self, x, mask=False):
        return self.student(x, mask=mask)

    @torch.inference_mode()
    def embed(self, x):
        return self.student(x)

def augment(x, include_identity=False):
    """Applies random rotation + reflection, avoiding double counting"""
    
    # construct options
    idcs = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    if include_identity:
        idcs.append((0,0))

    # select from options
    ref_idx, rot_idx = random.choice(idcs)

    # apply transformations
    x = torch.rot90(x, rot_idx, dims=[2,3])
    if ref_idx:
        x = x.transpose(2, 3)
    
    return x  