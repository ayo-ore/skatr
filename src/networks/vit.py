import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from functools import partial
from hydra.utils import instantiate
from torch.utils.checkpoint import checkpoint

from src.utils import masks
from src.utils.config import get_prev_config


class ViT(nn.Module):
    """
    A vision transformer network.
    """

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        self.patch_shape = cfg.patch_shape
        self.in_shape = cfg.in_shape
        in_channels, *axis_sizes = cfg.in_shape
        self.dim = cfg.hidden_dim

        if self.cfg.use_patching:

            # check consistency of arguments
            check_shapes(cfg)

            # embedding layer
            self.patch_dim = math.prod(cfg.patch_shape) * in_channels
            self.embedding = nn.Linear(self.patch_dim, self.dim)

        # position encoding
        fourier_dim = self.dim // 6  # sin/cos features for each dim
        w = torch.arange(fourier_dim) / (fourier_dim - 1)
        w = (1.0 / (10_000**w)).repeat(3)
        self.pos_encoding_freqs = nn.Parameter(
            w.log() if cfg.learn_pos_encoding else w,
            requires_grad=cfg.learn_pos_encoding,
        )
        self.init_pos_grid(axis_sizes)

        # transformer stack
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.dim,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    mlp_drop=cfg.mlp_drop,
                    checkpoint_grads=cfg.checkpoint_grads,
                    attn_drop=cfg.attn_drop,
                    proj_drop=cfg.proj_drop,
                    linear=cfg.linear,
                    # seq_len=math.prod(self.num_patches),
                    # linear_k=cfg.linear_k,
                )
                for _ in range(cfg.depth)
            ]
        )

        # norm layer
        self.out_norm = nn.LayerNorm(self.dim, eps=1e-6)

        # optionally initialize a task head, input pooling, or mask token
        if cfg.use_head:
            self.init_head(cfg.head)
        if self.cfg.use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(self.dim))

    def init_head(self, cfg):
        self.head = instantiate(cfg)

    def init_pos_grid(self, axis_sizes):
        self.num_patches = [s // p for s, p in zip(axis_sizes, self.cfg.patch_shape)]
        for i, n in enumerate(self.num_patches):  # axis values for each dim
            self.register_buffer(f"grid_{i}", torch.arange(n) * (2 * math.pi / n))

    def pos_encoding(self, scales=None):

        if not self.cfg.use_patching:
            return 0.0  # potentially dangerous

        grids = [
            getattr(self, f"grid_{i}") / (1 if scales is None else scales[i])
            for i in range(3)
        ]
        coords = torch.meshgrid(*grids, indexing="ij")

        if self.cfg.learn_pos_encoding:
            freqs = self.pos_encoding_freqs.exp().chunk(3)
        else:
            freqs = self.pos_encoding_freqs.chunk(3)

        features = [
            trig_fn(x.flatten()[:, None] * w[None, :])
            for (x, w) in zip(coords, freqs)
            for trig_fn in (torch.sin, torch.cos)
        ]
        return torch.cat(features, dim=1)

    def forward(self, x, mask=None, scales=None):
        """
        Forward pass of ViT.
        :param x   : tensor of spatial inputs with shape (batch_size, channels, *axis_sizes)
        :param mask: a tensor of patch indices that should be masked out of `x`.
        """

        if self.cfg.use_patching:
            # patchify input
            # x -> (batch_size, number_of_patches, voxels_per_patch)
            x = self.to_patches(x)

            # embed
            # x -> (batch_size, number_of_patches, embedding_dim)
            if hasattr(self, "extra_proj"):
                x = self.extra_proj(x)
            x = self.embedding(x)

            # apply mask and position encoding
            if self.cfg.use_mask_token:
                if mask is not None:
                    x = self.apply_mask_tokens(x, mask)
                x = x + self.pos_encoding(scales=scales)
            else:
                # x -> (batch_size, number_of_masked_patches, embedding_dim)
                x = x + self.pos_encoding(scales=scales)
                if mask is not None:
                    x = masks.gather_tokens(x, mask)

        # process patches with transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)

        if hasattr(self, "head"):
            # aggregate patch features and apply task head
            # x -> (batch_size, out_channels)
            x = torch.mean(x, axis=1)
            x = self.head(x)

        return x

    def to_patches(self, x):
        x = rearrange(
            x,
            "b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)",
            **dict(zip(("p1", "p2", "p3"), self.patch_shape)),
        )
        return x

    def apply_mask_tokens(self, x, mask_idcs):
        """
        Replaces patch embeddings in `x` with the network's mask token at indices speficied by `mask`.

        :param x   : input tensor with shape (B [batch size], T [number of patches], D [embed dim])
        :param mask: tensor with shape (B, T) containing indices in the range [0,T)
        """
        B, T = x.shape[:2]
        full_mask_token = repeat(self.mask_token, "d -> b t d", b=B, t=T)
        # construct boolean mask
        mask = torch.zeros((B, T), device=x.device).scatter_(-1, mask_idcs, 1).bool()
        return torch.where(mask[..., None], full_mask_token, x)


class PredictorViT(ViT):

    def __init__(self, cfg):

        super().__init__(cfg)

        # override embedding layer
        self.embedding = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.in_dim)

    def forward(self, ctx, ctx_mask, tgt_mask):
        """
        :param ctx: tokens from context block
        :param ctx_mask: mask corresponding to context block (for pos encoding)
        :param tgt_mask: mask corresponding to target block (for pos encoding)
        """

        B, N_ctx, D = ctx.shape  # batch size, num context patches, context dim
        T = math.prod(self.num_patches)  # total patches (before masking)

        pos_encoding = repeat(self.pos_encoding(), "t d -> b t d", b=B)

        # embed context tokens to own hidden dim
        ctx = self.embedding(ctx)
        ctx = ctx + masks.gather_tokens(
            pos_encoding, ctx_mask
        )  # TODO: Correct? or different pos encoding needed?

        # prepare target prediction tokens
        tgt = repeat(self.mask_token, "d -> b t d", b=B, t=T)  # repeat to full shape
        tgt = tgt + pos_encoding  # add position encodings
        tgt = masks.gather_tokens(tgt, tgt_mask)  # only keep tokens in target block

        # concatenate
        prd = torch.cat([ctx, tgt], dim=1)

        # process patches with transformer blocks
        for block in self.blocks:
            prd = block(prd)
        prd = self.out_norm(prd)

        prd = prd[:, N_ctx:]  # select output tokens in target block
        prd = self.out_proj(prd)  # project back to full dimensions

        return prd


class Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        mlp_drop=0.0,
        checkpoint_grads=False,
        linear=False,
        seq_len=None,
        linear_k=256,
        **attn_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = (LinearAttention if linear else Attention)(
            hidden_size, num_heads=num_heads, qkv_bias=True, **attn_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=mlp_drop,
        )
        # self.mlp = FeedForward(
        #     dim=hidden_size,
        #     mult=mlp_ratio,
        #     dropout=mlp_drop,
        #     glu=True,
        # )
        self.checkpoint_grads = checkpoint_grads

    def forward(self, x):
        if self.checkpoint_grads:
            x = x + checkpoint(self.attn, self.norm1(x), use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PretrainedViT(ViT):
    """
    A class for initializing pretrained ViTs.
    """

    def __init__(self, cfg):

        # read backbone config
        bb_dir = cfg.backbone_dir
        bcfg = get_prev_config(bb_dir)

        # load backbone state
        model_state = torch.load(os.path.join(bb_dir, "model.pt"))["model"]
        net_state = {
            k.replace("net.", ""): v
            for k, v in model_state.items()
            if k.startswith("net.")
        }

        # initialize network and load weights
        super().__init__(bcfg.net)
        self.load_state_dict(net_state)

        # delete the head module used in pretraining
        if cfg.drop_head and hasattr(self, "head"):
            del self.head

        # freeze weights and set to eval mode
        if cfg.frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        # init new head or input adaption if needed
        if cfg.add_head:
            self.head = instantiate(cfg.head)
        if cfg.adapt_res:
            self.init_adaptor(cfg.adaptor)
        if cfg.use_input_conv:
            self.init_input_conv(cfg.input_conv)
        if cfg.interp_pos_encoding:
            self.bb.init_pos_grid(cfg.data_shape)


def check_shapes(cfg):
    for i, (s, p) in enumerate(zip(cfg.in_shape[1:], cfg.patch_shape)):
        assert (
            not s % p
        ), f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
    image_dim = (len(cfg.in_shape) - 1)
    assert (
        not cfg.hidden_dim % 2*image_dim
    ), f"Hidden dim should be divisible by {2*image_dim} (for fourier position embeddings)"


class HiLoAdaptor(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        # load backbone vit
        self.vit = PretrainedViT(cfg.backbone)

        # get hi- and lo-res shapes
        self.hr_shape = cfg.in_shape[1:3]  # assuming no tiling in z direction
        self.lr_shape = self.vit.in_shape[1:3]

        # determine zooming in each axis
        self.zoom_factors = [s // v for s, v in zip(self.hr_shape, self.lr_shape)]
        # optionally rescale ViT positional encoding
        self.pos_encoding_scales = (
            None if not cfg.rescale_pos_encoding else self.zoom_factors + [1.0]
        )

        # One-hot positional encodings
        self.register_buffer("one_hot_eye", torch.eye(math.prod(self.zoom_factors) + 1))

        # Fixed sin/cos positional encodings
        fourier_dim = self.vit.dim // 4
        w = torch.arange(fourier_dim) / (fourier_dim - 1)
        self.register_buffer("pos_encoding_freqs", (1.0 / (10_000**w)).repeat(2))
        for i, n in enumerate(self.zoom_factors):
            self.register_buffer(f"grid_{i}", torch.arange(n) * (2 * math.pi / n))

        # TODO: allow for mismatched embedding dims
        # self.embedding = nn.Linear(self.vit.dim, self.dim)
        # assert self.vit.dim == self.dim

    def pos_encoding(self):
        coords = torch.meshgrid(self.grid_0, self.grid_1, indexing="ij")
        freqs = self.pos_encoding_freqs.chunk(2)
        features = [
            trig_fn(x.flatten()[:, None] * w[None, :])
            for (x, w) in zip(coords, freqs)
            for trig_fn in (torch.sin, torch.cos)
        ]

        return torch.cat(features, dim=1)

    def forward(self, x):

        batchsize = x.size(0)

        # downsample image
        x_global = reduce(
            x,
            "b c (nx px) (ny py) z -> b c nx ny z",
            "mean",
            **dict(zip(("px", "py"), self.zoom_factors)),
        )
        # split image into tiles
        xs_local = rearrange(
            x,
            "b c (nx px) (ny py) z -> b (nx ny) c px py z",
            **dict(zip(("px", "py"), self.lr_shape)),
        )

        # forward pass all through vit and stack
        z_global = self.vit(x_global).mean(-2)
        zs_local = (
            self.vit(xs_local.flatten(0, 1), scales=self.pos_encoding_scales)
            .unflatten(0, (batchsize, -1))
            .mean(-2)
        )

        zs_local = zs_local + self.pos_encoding()  # .unsqueeze(0)
        z = torch.cat([z_global.unsqueeze(1), zs_local], 1)

        # pos_encoding = self.one_hot_eye[None, ...].repeat(batchsize, 1, 1) # old
        # z = torch.cat([z, pos_encoding], 2) # old

        return z


class HiLoViT(ViT):

    def __init__(self, cfg):

        super().__init__(cfg)

        # load backbone vit
        self.vit = PretrainedViT(cfg.backbone)

        # get hi- and lo-res shapes
        self.hr_shape = cfg.in_shape[1:3]  # assuming no tiling in z direction
        self.lr_shape = self.vit.in_shape[1:3]

        # determine zooming in each axis
        self.zoom_factors = [s // v for s, v in zip(self.hr_shape, self.lr_shape)]

        # optionally rescale ViT positional encoding
        if cfg.rescale_pos_encoding:
            self.vit.grid_0 /= self.zoom_factors[0]
            self.vit.grid_1 /= self.zoom_factors[1]

        # initialize unstructured positional encodings for local tiles
        self.pos_encoding_mtx = nn.Parameter(
            torch.empty(math.prod(self.zoom_factors) + 1, self.dim)
        )
        torch.nn.init.trunc_normal_(self.pos_encoding_mtx)

        # TODO: allow for mismatched embedding dims
        # self.embedding = nn.Linear(self.vit.dim, self.dim)
        assert self.vit.dim == self.dim

    @property
    def pos_encoding(self):
        return self.pos_encoding_mtx

    def forward(self, x):

        batchsize = x.size(0)

        # downsample image
        x_global = reduce(
            x,
            "b c (nx px) (ny py) z -> b c nx ny z",
            "mean",
            **dict(zip(("px", "py"), self.zoom_factors)),
        )
        # split image into tiles
        xs_local = rearrange(
            x,
            "b c (nx px) (ny py) z -> b (nx ny) c px py z",
            **dict(zip(("px", "py"), self.lr_shape)),
        )

        # forward pass all through vit and stack
        z_global = self.vit(x_global).mean(-2)
        zs_local = (
            self.vit(xs_local.flatten(0, 1)).unflatten(0, (batchsize, -1)).mean(-2)
        )
        z = torch.cat([z_global.unsqueeze(1), zs_local], 1)

        # add pos encoding
        z = z + self.pos_encoding

        for block in self.blocks:
            z = block(z)

        z = self.out_norm(z)

        if hasattr(self, "head"):
            # aggregate patch features and apply task head
            # z -> (batch_size, out_channels)
            z = torch.mean(z, axis=1)
            z = self.head(z)

        return z


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
    ):
        super().__init__()
        assert (dim % heads) == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert (
            kv_len <= self.seq_len
        ), f"the sequence length of the key / values must be {self.seq_len} - {kv_len} given"

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum("bnd,nk->bkd", *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections

        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = (
            lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        )
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (d_h**-0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class LinearAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # 3, B, H, N, C
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale
        # v = l2norm(v)

        # k, v = map(lambda t: t / math.sqrt(self.seq_len), (k, v))

        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        out = torch.einsum("b h n d, b h d e -> b h n e", q, context)
        out = rearrange(out, "b h n c -> b n (h c)", h=self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, int(dim * mult) * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(int(dim * mult), dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
