import torch
import numpy as np
from hydra.utils import instantiate

from .config import get_prev_config


def load_model(path, model_cls=None, device=None, freeze=True):

    cfg = get_prev_config(path)

    # TODO: tidy this to just use instantiate
    model = model_cls(cfg) if model_cls is not None else instantiate(cfg.model, cfg=cfg)
    # model = instantiate(cfg.model, cfg=cfg)
    sdict = torch.load(path + "/model.pt", weights_only=False, map_location="cpu")

    # load (and optionally freeze) weights
    model.load_state_dict(sdict["model"])
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    # set to eval mode (disable batchnorm, dropout etc.)
    model.eval()

    # move to device
    if device is not None:
        model = model.to(device)

    return model, cfg

def ensure_device(x, device):
    """Recursively send tensors within nested structure to device"""
    if isinstance(x, list):
        return [ensure_device(e, device) for e in x]
    if isinstance(x, tuple):
        return tuple(ensure_device(e, device) for e in x)
    elif x.device != device:
        return x.to(device=device, non_blocking=True)
    return x
        