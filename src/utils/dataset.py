import logging
import numpy as np
import os
import torch

from glob import glob
from functools import partial
from tensordict import MemoryMappedTensor, tensorclass
from torch import nn
from torch.utils.data import DataLoader
from tqdm.contrib.concurrent import thread_map
from typing import Optional

from src.utils.augmentations import RotateAndReflect
from src.utils.collators import SummarizationCollator
from src.utils.utils import ensure_device

KEYS = (
    "image",  # keys present in your npz files, you could add more...
    "label",
)

DTYPES = {
    "image": torch.float32,  # can set as you please, but probably best to keep f32
    "label": torch.float32,
}

log = logging.getLogger(__name__)


@tensorclass
class LightconeData:
    """
    'images', shape (N_data, x, y, z): 21cm maps
    'labels', shape (N_data, N_params): parameter vector
    """

    images: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    summaries: Optional[torch.Tensor] = None

    @classmethod
    def read(
        cls,
        path: str,
        shapes: dict,
        num_workers: int = 4,
        summary_cfg: Optional[dict] = None,
    ):
        """
        Read lightcone data as memory-mapped tensors. If the memory map does not yet exist, it will
        be created.

        Args:
        path: Directory containing lightcones as separate .npz files.
        num_workers: Number of processes to use when writing the memmap. Ignored if memmap exists.
        """

        # find all numpy records
        files = glob(os.path.join(path, "run*.npz"))
        size = len(files)

        # iterate keys and fill tensors
        tensors = {}
        for k in KEYS:

            ks = k + "s"  # plural naming is nicer

            filename = os.path.join(path, ks + ".memmap")
            dtype = DTYPES[k]
            shape = (size, *shapes[k])
            if os.path.exists(filename):  # read memmap from disk

                log.info(f"Reading memmap '{os.path.basename(filename)}'")
                mmap = MemoryMappedTensor.from_filename(
                    filename=filename,
                    dtype=dtype,
                    shape=shape,
                )

                summarize = summary_cfg is not None
                if summarize and (ks == "images"):

                    log.info(
                        "Summarizing lightcones"
                        + (
                            " (with augmentations)"
                            if summary_cfg["augmentations"]
                            else ""
                        )
                    )

                    # create dataloader
                    loader = DataLoader(
                        mmap,
                        batch_size=summary_cfg["batch_size"],
                        num_workers=num_workers,
                        collate_fn=SummarizationCollator(
                            preprocessing=summary_cfg["preprocessing"], training=False
                        ),
                    )

                    device = torch.device(summary_cfg["device"])
                    summary_net = summary_cfg["net"]

                    # summarize lightcones
                    summaries = []
                    for batch in loader:

                        batch = ensure_device(batch, device)
                        with torch.no_grad(), torch.autocast(
                            device.type, enabled=summary_cfg["use_amp"]
                        ):
                            # embed with pretrained net
                            if not summary_cfg["augment"]:
                                summary = summary_net(batch).to(device)
                            else:
                                aug = RotateAndReflect(include_identity=True)
                                summary = torch.stack(  # collect all augmentations of lightcones
                                    [
                                        summary_net(abatch).to(device)
                                        for abatch in aug.enumerate(batch)
                                    ],
                                    dim=1,
                                )

                            if summary_cfg["pool"]:
                                # one summary per lightcone
                                summary = summary.mean(-2)
                            summaries.append(summary)

                    summaries = torch.vstack(summaries).cpu()

                    tensors["summaries"] = summaries

                    log.info("Finished summarizing lightcones")

                else:
                    tensors[ks] = mmap

            else:  # or write new memmap to disk

                log.info(f"Writing memmap '{os.path.basename(filename)}'")

                tensors[ks] = MemoryMappedTensor.empty(  # placeholder
                    filename=filename,
                    dtype=dtype,
                    shape=shape,
                )

                thread_map(  # fill in parallel
                    partial(worker_func, key=k, files=files, tensors=tensors),
                    range(size),
                    max_workers=4,
                )

        # return tensorclass dataset
        return cls(batch_size=[size], **tensors)


# worker function for parallel processing
def worker_func(i, key, files, tensors):
    arr = np.load(files[i])[key]
    tensors[key + "s"][i] = torch.from_numpy(arr)


# class SummarizedLightconeDataset(Dataset):

#     def __init__(
#         self, dataset, summary_net, device, exp_cfg, dataset_cfg, augment=False, use_amp=False
#     ):

#         self.Xs = []
#         self.ys = []
#         self.summary_net = summary_net

#         self.pool_summary = not (
#             hasattr(summary_net, "head")
#             or exp_cfg.net.arch == "AttentiveHead"
#             or (hasattr(exp_cfg, "use_attn_pool") and exp_cfg.use_attn_pool)
#         )

#         if augment:
#             aug = RotateAndReflect(include_identity=True)

#         dataloader = DataLoader(
#             dataset,
#             batch_size=dataset_cfg.summary_batch_size,
#             num_workers=exp_cfg.num_cpus if exp_cfg.num_cpus > 1 else 0,
#         )

#         dset_device = device if dataset_cfg.on_gpu else torch.device("cpu")
#         for X, y in dataloader:

#             self.ys.append(y.to(dset_device))

#             X = X.to(device)
#             with torch.autocast(device.type, enabled=use_amp):
#                 if augment:
#                     summary = torch.stack(  # collect all augmentations of X
#                         [self.summarize(xa).to(dset_device) for xa in aug.enumerate(X)],
#                         dim=1,
#                     )
#                 else:
#                     summary = self.summarize(X).to(dset_device)
#             self.Xs.append(summary)

#         self.Xs = torch.vstack(self.Xs)
#         self.ys = torch.vstack(self.ys)

#     @torch.no_grad()
#     def summarize(self, x):
#         x = self.summary_net(x)
#         if self.pool_summary:
#             x = x.mean(1)  # (B, T, D) --> (B, D)
#         return x

#     def __len__(self):
#         return len(self.Xs)

#     def __getitem__(self, idx):
#         return self.Xs[idx], self.ys[idx]

#     def collate_fn(self, batch):
#         """A collator that selects one augmentation of X."""
#         # Same augmentation for each batch element. Alternative is to append in a loop
#         X, y = torch.utils.data.default_collate(batch)
#         idx = torch.randint(X.size(1), ())
#         return X[:, idx], y
