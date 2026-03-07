import torch

from src import models
from src.experiments.training import TrainingExperiment
from src.utils.dataset import LightconeData
from src.utils.masks import jepa_mask

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
    
    def collate_fn(self, batch):
        """Perform preprocessing and masking on CPU. Avoids GPU sync during training."""

        # preprocess images
        for transform in self.preprocessing["x"]:
            batch.images = transform.forward(batch.images)

        num_patches = self.model.net.num_patches
        
        # sample masks
        tgt_masks, ctx_masks = jepa_mask(
            num_patches, self.cfg.masking, batch_size=len(batch), device=batch.device
        )

        return batch.images, tgt_masks, ctx_masks
    
    @torch.inference_mode()
    def evaluate(self, dataloaders):
        pass
    
    def plot(self):
        pass    
