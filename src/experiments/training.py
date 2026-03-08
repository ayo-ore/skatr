import torch
import os
import numpy as np
from abc import abstractmethod
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split

from src.experiments.base_experiment import BaseExperiment
from src.utils.dataset import LightconeData
from src.utils.trainer import Trainer
from src.utils import augmentations


class TrainingExperiment(BaseExperiment):

    def run(self):

        # initialize preprocessing transforms (for data and targets)
        self.preprocessing = {
            k: [instantiate(t) for t in ts] for k, ts in self.cfg.preprocessing.items()
        }
        transform_names = {
            k: [t.__class__.__name__ for t in ts]
            for k, ts in self.preprocessing.items()
        }
        self.log.info(f"Loaded preprocessing dict: {transform_names}")

        self.augmentations = self.get_augmentations()
        if self.augmentations:
            self.log.info(
                f"Loaded augmentations: {', '.join([a.__class__.__name__ for a in self.augmentations])}"
            )

        if self.cfg.train:

            # model
            if not hasattr(self, "model"):
                self.init_model()

            # dataloaders
            self.log.info("Creating dataLoaders")
            self.dataloaders = dict(
                zip(("train", "val", "test"), self.init_dataloader(training=True))
            )

            # train model
            self.log.info("Running training")

            trainer = Trainer(
                model=self.model,
                dataloaders=self.dataloaders,
                cfg=self.cfg.training,
                exp_dir=self.exp_dir,
                device=self.device,
                use_amp=self.cfg.use_amp,
            )
            trainer.run_training()

        # evaluate model
        if self.cfg.evaluate:

            # model
            if not hasattr(self, "model"):
                self.init_model()

            if not hasattr(self, "dataloaders"):
                self.log.info("Creating dataLoaders")
                self.dataloaders = dict(
                    zip(
                        ("train", "val", "test"), self.init_dataloader(training=False)
                    )  # TODO: depracate training argument
                )

            # load model state
            self.log.info(f"Loading model state from {self.exp_dir}.")
            self.model.load(self.exp_dir, self.device)
            self.model.eval()

            self.log.info("Running evaluation on test dataset")
            self.evaluate(self.dataloaders["test"])

            del self.dataloaders
            torch.cuda.empty_cache()

        # make plots
        if self.cfg.plot:
            self.log.info("Making plots")
            self.plot()

        # print memory usage
        self.log_resources()

    def init_model(self):
        self.log.info("Initializing model")

        # TODO address further summarization conditions: could also be backbone finetuning

        if self.supervised:
            summarize = not ((self.cfg.summary_net is None) or self.cfg.data.summarize)
            summarize_kwarg = {"summarize": summarize}
        else:
            summarize_kwarg = {}

        self.model = instantiate(self.cfg.model, **summarize_kwarg)

        self.model = self.model.to(self.device)
        model_name = (
            f"{self.model.__class__.__name__}[{self.model.net.__class__.__name__}]"
        )
        num_params = sum(w.numel() for w in self.model.trainable_parameters)
        self.log.info(f"Model ({model_name}) has {num_params} trainable parameters")

    def init_dataloader(self, training=False):

        dcfg = self.cfg.data
        tcfg = self.cfg.training
        dscfg = self.cfg.dataset

        if dcfg.summarize:
            summary_cfg = {
                "net": self.model.summary_net,
                "batch_size": dcfg.summary_batch_size,
                "pool": True,  # TODO: Change pooling logic for AttentiveHead
                "preprocessing": self.preprocessing,
                "augmentations": self.augmentations if tcfg.augment else None,
                "device": self.device,
                "use_amp": self.cfg.training.use_amp,
            }
        else:
            summary_cfg = None

        # read data
        dset = LightconeData.read(
            dscfg.dir,
            shapes=dict(label=[dscfg.num_params], image=dscfg.image_shape),
            num_workers=dcfg.num_workers,
            summary_cfg=summary_cfg,
        )

        # TODO: free summary net from memory

        # optionally move dataset to gpu
        on_gpu = dcfg.on_gpu and self.cfg.use_gpu
        if on_gpu:
            dset = dset.to(self.device)

        # split dataset
        dsets = self.split_dataset(dset)

        self.log.info(f"Read dataset:\n{dset}")

        # create dataloaders
        dataloaders = []
        num_workers = 0 if on_gpu or not self.cfg.train else max(dcfg.num_workers, 0)
        use_mp = self.cfg.train and num_workers > 0
        for i, d in enumerate(dsets):

            is_train_split = (i == 0) and training
            batch_size = tcfg.batch_size if is_train_split else tcfg.test_batch_size

            dataloaders.append(
                DataLoader(
                    d,
                    shuffle=is_train_split,
                    drop_last=is_train_split,
                    batch_size=batch_size,
                    collate_fn=self.get_collator(training=is_train_split),
                    num_workers=num_workers,
                    pin_memory=self.cfg.use_gpu and not on_gpu,
                    multiprocessing_context="spawn" if use_mp else None,
                    persistent_workers=use_mp,
                )
            )

        return dataloaders

    def split_dataset(self, dset):

        dcfg = self.cfg.data

        # create splits
        assert dcfg.val_frac > 0, "A validation split is required"
        assert dcfg.test_frac > 0, "A testing split is required"

        # seed data split to avoid leakage across iterations
        fixed_rng = torch.Generator().manual_seed(1729)
        splits = random_split(
            dset,
            [1 - dcfg.val_frac - dcfg.test_frac, dcfg.val_frac, dcfg.test_frac],
            generator=fixed_rng,
        )

        return list(splits)

    def get_augmentations(self):
        augs = []
        if self.cfg.training.augment and not self.cfg.data.summarize:
            for name, kwargs in self.cfg.training.augmentations.items():
                aug = getattr(augmentations, name)(**kwargs)
                augs.append(aug)
        return augs

    def get_collator(self):
        """Perform experiment-specific collation. Can help to avoid CPU-GPU sync during training."""
        return IdentityCollator()

    @abstractmethod
    def init_dataset(self, path_exp, path_sim):
        """Read and return a dataset. To be implemented by the child class"""
        pass

    @abstractmethod
    def evaluate(self):
        """Iterate dataset and save model predictions. To be implemented by the child class"""
        pass

    @abstractmethod
    def plot(self):
        """Create and save evaluation plots. To be implemented by the child class"""
        pass


# collate_fn = default_collate
# is_trn = k=='train'
# # optionally summarize (compress) dataset
# if dcfg.summarize:

#     if self.cfg.summary_net is None:
#         self.log.error('Asking to summarize dataset, but no summary net provided.')
#         sys.exit()

#     augment = tcfg.augment and is_trn # only augment training split
#     augstring = ' (with augmentations) ' if augment else ' '
#     self.log.info(
#         f'Summarizing {k} split{augstring}with batch size {dcfg.summary_batch_size}.'
#     )

#     dataset_splits[k] = SummarizedLCDataset(
#         d, summary_net=self.model.summary_net, device=self.device, exp_cfg=self.cfg,
#         dataset_cfg=dcfg, augment=augment, use_amp=tcfg.use_amp
#     )

#     if augment:
#         # select one augmentation per batch
#         collate_fn = dataset_splits[k].collate_fn
