import os
import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig

from .. import networks

log = logging.getLogger(__name__)

class Model(nn.Module):

    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
        try:
            self.net = getattr(networks, cfg.net.arch)(cfg.net)
        except AttributeError as e:
            log.error(f'Network architecture "{cfg.net.arch}" not recognized!')
            raise e

    def load(self):
        path = os.path.join(self.cfg.exp_dir, 'model.pt')
        state_dicts = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state_dicts["net"])

    def save(self):
        raise NotImplementedError
