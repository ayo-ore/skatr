import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List

log = logging.getLogger("Trainer")


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        preprocessing: Dict[str, List[Callable]],
        augmentations: List[Callable],
        cfg: DictConfig,
        exp_dir: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        model           -- a pytorch model to be trained
        dataloaders     -- a dictionary containing pytorch data loaders at keys 'train' and 'val'
        preprocessing   -- A dictinoary containing a list of transformations for each element in
                           the batch tuple. Processing is applied at each training iteration
        cfg             -- configuration dictionary
        exp_dir         -- directory to which training outputs will be saved
        """

        self.model = model
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.exp_dir = exp_dir
        self.preprocessing = preprocessing
        self.augmentations = augmentations
        self.start_epoch = 0
        self.patience_counter = 0

    def prepare_training(self):

        log.info("Preparing model training")

        # init optimizer
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        self.optimizer = opt_cls(
            self.model.trainable_parameters, lr=self.cfg.lr, **self.cfg.optimizer.kwargs
        )

        # init scheduler
        self.steps_per_epoch = len(self.dataloaders["train"])
        if self.cfg.scheduler:
            self.scheduler = self.init_scheduler()

        # set logging of metrics
        if self.cfg.use_tensorboard:
            self.summarizer = SummaryWriter(self.exp_dir)
            log.info(f"Writing tensorboard summaries to dir {self.exp_dir}")
        else:
            log.info("`use_tensorboard` set to False. No summaries will be written")

        self.epoch_train_losses = np.array([])
        self.epoch_val_losses = np.array([])
        self.best_val_loss = np.inf

        if self.cfg.warm_start:
            checkpoint = f"model{self.cfg.warm_start_epoch or ''}.pt"
            path = os.path.join(self.exp_dir, checkpoint)
            self.load(path)
            # avoid overriding checkpoint
            os.rename(path, path.replace(".pt", f"_old.pt"))
            log.info(f"Warm starting training from epoch {self.start_epoch}")

    def run_training(self):

        self.prepare_training()

        num_epochs = self.cfg.epochs - self.start_epoch
        log.info(f"Beginning training loop with epochs set to {num_epochs}")
        if self.cfg.patience:
            log.info(f"Early stopping patience set to {self.cfg.patience}")

        t_0 = time.time()
        for e in range(num_epochs):

            t0 = time.time()
            self.epoch = (self.start_epoch or 0) + e

            # train
            self.model.net.train()
            self.train_one_epoch()

            # validate at given frequency
            if (self.epoch + 1) % self.cfg.validate_freq == 0:
                self.model.eval()
                self.validate_one_epoch()

                # check whether validation loss improved
                if (val_loss := self.epoch_val_losses[-1]) < self.best_val_loss:
                    self.patience_counter = 0

                    if self.cfg.save_best_epoch:  # save best checkpoint
                        self.best_val_loss = val_loss
                        self.save()

                elif self.cfg.patience:  # early stopping
                    self.patience_counter += 1
                    if self.patience_counter == self.cfg.patience:
                        log.info(f"Stopping training early at epoch {self.epoch}")
                        break

            # optionally save model at given frequency
            if save_freq := self.cfg.save_freq:
                if (self.epoch + 1) % save_freq == 0 or self.epoch == 0:
                    self.save(tag=self.epoch)

            # estimate training time
            if e == 0:
                t1 = time.time()
                dtEst = (t1 - t0) * num_epochs
                log.info(
                    f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h"
                )

        t_1 = time.time()
        traintime = t_1 - t_0
        log.info(
            f"Finished training {self.epoch + 1} epochs after {traintime:.2f} s"
            f" = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h."
        )

        # save final model
        if not self.cfg.save_best_epoch:
            log.info("Saving final model")
            self.model.eval()
            self.save()

    def train_one_epoch(self):

        # create list to save loss per iteration
        train_losses = []

        # iterate batch wise over input
        for itr, batch in enumerate(self.dataloaders["train"]):

            # clear optimizer gradients
            self.optimizer.zero_grad(set_to_none=True)

            # place batch on device
            batch = ensure_device_and_dtype(batch, self.device, self.dtype)

            # augment
            for augment in self.augmentations:
                batch[0] = augment(batch[0])

            # calculate batch loss
            loss = self.model.batch_loss(batch)

            # update model parameters
            step = itr + self.epoch * self.steps_per_epoch
            total_steps = self.cfg.epochs * self.steps_per_epoch
            self.model.update(self.optimizer, loss, step, total_steps)

            # update learning rate
            if self.cfg.scheduler:
                self.scheduler.step()

            # track iter loss
            train_losses.append(loss.detach())
            log_step = self.cfg.log_iters
            if self.cfg.use_tensorboard and (not step % log_step) or not step:
                self.summarizer.add_scalar(
                    "iter_loss_train",
                    torch.stack(train_losses[-log_step:]).mean().cpu().numpy(),
                    step,
                )

        # track epoch loss
        self.epoch_train_losses = np.append(
            self.epoch_train_losses, torch.stack(train_losses).mean().cpu().numpy()
        )

        # optionally log to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar(
                "epoch_loss_train", self.epoch_train_losses[-1], self.epoch
            )
            if self.cfg.scheduler:
                self.summarizer.add_scalar(
                    "learning_rate", self.scheduler.get_last_lr()[0], self.epoch
                )

    @torch.inference_mode()
    def validate_one_epoch(self):

        # calculate loss batchwise over input
        val_losses = []
        for batch in self.dataloaders["val"]:

            # augment
            if self.cfg.augment_test:
                for augment in self.augmentations:
                    batch[0] = augment(batch[0])

            # place x on device
            batch = ensure_device_and_dtype(batch, self.device, self.dtype)
            # calculate loss
            loss = self.model.batch_loss(batch).detach()
            val_losses.append(loss)

        # track loss
        self.epoch_val_losses = np.append(
            self.epoch_val_losses, torch.stack(val_losses).mean().cpu().numpy()
        )

        # optional logging to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar(
                "epoch_loss_val", self.epoch_val_losses[-1], self.epoch
            )

    def save(self, tag=""):
        """Save the model along with the training state"""
        state_dicts = {
            "opt": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "losses": self.epoch_train_losses,
            "epoch": self.epoch,
        }
        if self.cfg.scheduler:
            state_dicts["scheduler"] = self.scheduler.state_dict()
        torch.save(state_dicts, os.path.join(self.exp_dir, f"model{tag}.pt"))

    def load(self, path):
        """Load the model and training state"""

        state_dicts = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dicts["model"])
        if "losses" in state_dicts:
            self.epoch_train_losses = state_dicts.get("losses", {})
        if "epoch" in state_dicts:
            self.start_epoch = state_dicts.get("epoch", 0) + 1
        if "opt" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["opt"])
        if "scheduler" in state_dicts:
            self.scheduler.load_state_dict(state_dicts["scheduler"])
        self.model.net.to(self.device)

    def init_scheduler(self):
        name = self.cfg.scheduler.name
        sdl_cls = getattr(torch.optim.lr_scheduler, name)
        total_steps = self.cfg.epochs * self.steps_per_epoch
        match name:
            case "OneCycleLR":
                return sdl_cls(
                    self.optimizer, total_steps=total_steps, **self.cfg.scheduler.kwargs
                )
            case _:
                return sdl_cls(
                    self.optimizer, total_iters=total_steps, **self.cfg.scheduler.kwargs
                )


def ensure_device_and_dtype(x, device, dtype):
    """Recursively send tensors within nested structure to device"""
    if isinstance(x, list):
        return [ensure_device_and_dtype(e, device, dtype) for e in x]
    if isinstance(x, tuple):
        return tuple(ensure_device_and_dtype(e, device, dtype) for e in x)
    else:
        if x.device == device and x.dtype == dtype:
            return x
        elif x.dtype == dtype:
            return x.to(device)
        elif x.device == device:
            return x.to(dtype)
        else:
            return x.to(device=device, dtype=dtype)
