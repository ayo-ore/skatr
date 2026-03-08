import torch

from src.experiments.training import TrainingExperiment
from src.utils.collators import JEPACollator


class PretrainingExperiment(TrainingExperiment):

    supervised: bool = False

    def get_collator(self, training):
        """Perform preprocessing and masking on CPU. Avoids GPU sync during training."""

        num_patches = tuple(
            s // p
            for s, p in zip(self.cfg.dataset.image_shape, self.cfg.net.patch_shape)
        )

        mask_cfg = self.cfg.masking

        return JEPACollator(self.preprocessing, num_patches, mask_cfg)

    @torch.inference_mode()
    def evaluate(self, dataloaders):
        pass

    def plot(self):
        pass
