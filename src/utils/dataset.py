import numpy as np
import os
import torch
from glob import glob
from functools import partial
from tqdm.contrib.concurrent import process_map
from tensordict import MemoryMappedTensor, tensorclass

KEYS = (
    "image",  # keys present in your npz files, you could add more...
    "label",
)

DTYPES = {
    "image": torch.float32,  # can set as you please, but probably best to keep f32
    "label": torch.float32,
}

@tensorclass
class LightconeData:
    """
    'images', shape (N_data, x, y, z): 21cm maps
    'labels', shape (N_data, N_params): parameter vector
    """

    images: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def from_memmap(
        cls,
        path: str,
        shapes: dict,
        num_workers: int = 4,
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

            ks = k + "s" # plural naming is nicer

            filename = os.path.join(path, ks + ".memmap")
            dtype = DTYPES[k]
            shape = (size, *shapes[k])
            if os.path.exists(filename):  # read memmap from disk

                print(f"Reading memmap '{os.path.basename(filename)}' from disk")
                tensors[ks] = MemoryMappedTensor.from_filename(
                    filename=filename,
                    dtype=dtype,
                    shape=shape,
                )
            else:  # or write new memmap to disk

                print(f"Writing memmap '{os.path.basename(filename)}' to disk")

                tensors[ks] = MemoryMappedTensor.empty(  # placeholder
                    filename=filename,
                    dtype=dtype,
                    shape=shape,
                )

                process_map(  # fill in parallel
                    partial(worker_func, key=k, files=files, tensors=tensors),
                    range(size),
                    max_workers=num_workers,
                    chunksize=1,
                )

        # return tensorclass dataset
        return cls(batch_size=[size], **tensors)


# worker function for parallel processing
def worker_func(i, key, files, tensors):
    arr = np.load(files[i])[key]
    tensors[key+"s"][i] = torch.from_numpy(arr)


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
