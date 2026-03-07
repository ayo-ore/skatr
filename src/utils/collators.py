from .dataset import LightconeData
from .masks import jepa_mask


class IdentityCollator:

    def __init__(self, training=True):
        self.training = training

    def __call__(batch: LightconeData):
        return batch


class SupervisedCollator:

    def __init__(self, preprocessing, augmentations, training=True):
        self.preprocessing = preprocessing
        self.augmentations = augmentations
        self.training = training

    def __call__(self, batch: LightconeData):

        for transform in self.preprocessing["x"]:
            batch.images = transform.forward(batch.images)

        for transform in self.preprocessing["y"]:
            batch.labels = transform.forward(batch.labels)

        if self.training:
            for aug in self.augmentations:
                batch.images = aug(batch.images)

        return batch


class JEPACollator:

    def __init__(self, preprocessing, num_patches, mask_cfg, training=True):
        self.preprocessing = preprocessing
        self.num_patches = num_patches
        self.mask_cfg = mask_cfg
        self.training = training

    def __call__(self, batch: LightconeData):

        # preprocess images
        for transform in self.preprocessing["x"]:
            batch.images = transform.forward(batch.images)

        # sample masks
        tgt_masks, ctx_masks = jepa_mask(
            self.num_patches,
            self.mask_cfg,
            batch_size=len(batch),
            device=batch.device,
        )

        return batch.images, tgt_masks, ctx_masks
