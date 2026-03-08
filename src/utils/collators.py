from .masks import jepa_mask
from torch.utils.data import default_collate


class IdentityCollator:

    def __init__(self, training=True):
        self.training = training

    def __call__(batch):
        return batch


class SupervisedCollator:

    def __init__(self, preprocessing, augmentations, training=True):
        self.preprocessing = preprocessing
        self.augmentations = augmentations
        self.training = training

    def __call__(self, batch):

        for transform in self.preprocessing["y"]:
            batch.labels = transform.forward(batch.labels)

        if batch.images is not None:
            for transform in self.preprocessing["x"]:
                batch.images = transform.forward(batch.images)

            if self.training:
                for aug in self.augmentations:
                    batch.images = aug(batch.images)

        if (batch.summaries is not None) and self.augmentations:
            # select a random augmentation
            idx = torch.randint(batch.summaries.size(1), ())
            batch.summaries = batch.summaries[:, idx]

        return batch


class SummarizationCollator:

    def __init__(self, preprocessing, training=True):
        self.preprocessing = preprocessing
        self.training = training

    def __call__(self, images):

        images = default_collate(images)
        # preprocess
        for transform in self.preprocessing["x"]:
            images = transform.forward(images)

        return images


class JEPACollator:

    def __init__(self, preprocessing, num_patches, mask_cfg, training=True):
        self.preprocessing = preprocessing
        self.num_patches = num_patches
        self.mask_cfg = mask_cfg
        self.training = training

    def __call__(self, batch):

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
