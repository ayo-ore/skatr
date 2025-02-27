import os
import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig

from src import networks
from src.utils.config import get_prev_config

log = logging.getLogger("Model")


class Model(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # optionally initialize a summary network
        if cfg.summary_net is not None:
            log.info("Loading summary network")
            sum_net_cls = getattr(networks, cfg.summary_net.arch)
            self.summary_net = sum_net_cls(cfg.summary_net)
            log.info(
                f"Summary net ({self.summary_net.__class__.__name__}) has "
                f"{sum(w.numel() for w in self.summary_net.parameters())} parameters"
            )

        # initialize network
        # TODO: depracate cfg.replace_backbone
        # TODO: Automatically set MLP input dim to backbone embedding dim
        net_cls = getattr(networks, cfg.net.arch)
        self.net = net_cls(cfg.net)

    @property
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def update(
        self, loss, optimizer, scaler, step=None, total_steps=None, summary_writer=None
    ):

        # scale gradients for mixed precision stability
        loss = scaler.scale(loss)

        # propagate gradients
        loss.backward()

        # optionally clip gradients
        if clip := self.cfg.training.gradient_norm:
            nn.utils.clip_grad_norm_(self.trainable_parameters, clip)
            # grad_norm = nn.utils.clip_grad_norm_(self.trainable_parameters, clip)
            # summary_writer.add_scalar("gradient_norm", grad_norm.cpu().item(), step)

        # update weights
        scaler.step(optimizer)
        scaler.update()

        # zero parameter gradients
        optimizer.zero_grad(set_to_none=True)

    def load(self, exp_dir, device):
        path = os.path.join(exp_dir, "model.pt")
        state_dicts = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state_dicts["model"])
