import torch


class AddSingletonChannel:
    """Add single channel dimension to input. Assumes three spatial dimensions"""

    def forward(self, x):
        return x.unsqueeze(-4)

    def reverse(self, x):
        return x.squeeze(-4)


class Center:
    """Shift and scale a tensor into the range [0,1] given min value `lo` and max value `hi`"""

    def __init__(self, lo, hi, indices=None, dtype=torch.float32):
        self.lo = torch.tensor(lo, dtype=dtype)
        self.hi = torch.tensor(hi, dtype=dtype)
        self.indices = indices
        if indices is not None:
            self.lo = self.lo[indices]
            self.hi = self.hi[indices]

    def forward(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)
        if self.indices is not None:
            x = x[..., sorted(self.indices)]
        return (x - self.lo) / (self.hi - self.lo)

    def reverse(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)
        return x * (self.hi - self.lo) + self.lo


class Clamp:
    """Apply a symmetric log scaling to the input."""

    def forward(self, x):
        return x.abs().add(1).log() * x.sign()

    def reverse(self, x):
        return x.abs().exp().add(-1) * x.sign()


class Upsample:
    """TODO: Fill docstring"""

    def __init__(self, factor):
        self.upsampler = torch.nn.Upsample(scale_factor=factor, mode="trilinear")
        self.downsampler = torch.nn.AvgPool3d(kernel_size=factor, stride=factor)

    def forward(self, x):
        return self.upsampler(x.unsqueeze(0)).squeeze(0)

    def reverse(self, x):
        return self.downsampler(x)
