import torch

from src.experiments.training import TrainingExperiment
from src.utils.collators import JEPACollator


class PretrainingExperiment(TrainingExperiment):

    # def get_dataset(self, directory):
    #     prep = self.preprocessing
    #     if self.cfg.data.file_by_file:
    #         return LightconeDatasetByFile(
    #             self.cfg.data, directory, preprocessing=prep, use_labels=False
    #         )
    #     else:
    #         return LightconeDataset(
    #             self.cfg.data,
    #             directory,
    #             self.device,
    #             preprocessing=prep,
    #             use_labels=False,
    #         )

    # def get_model(self):
    #     model_cls = getattr(models, self.cfg.model)
    #     return model_cls(self.cfg)

    def get_collator(self):
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
