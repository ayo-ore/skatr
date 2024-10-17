import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
# from hydra.utils import instantiate
from torch.utils.checkpoint import checkpoint
from itertools import pairwise

MAX_INPUT_ID = int(1e9)

from transformers import PretrainedConfig

# from src.utils import masks
# from src.utils.config import get_prev_config    

class ViT(nn.Module):
    """
    A vision transformer network.
    """

    def __init__(
        self,
        patch_shape=[4, 4, 10],
        in_shape=[1, 28, 28, 470],
        hidden_dim=144,
        learn_pos_encoding=True,
        num_heads=4,
        mlp_ratio=2.0,
        mlp_drop=0.,
        checkpoint_grads=False,
        attn_drop=0.,
        proj_drop=0.,
        depth=4,
        use_mask_token=True
    ):
        super().__init__()

        self.patch_shape = patch_shape
        in_channels, *axis_sizes = in_shape
        dim = hidden_dim
        
        # embedding layer
        self.patch_dim = math.prod(patch_shape) * in_channels
        self.embedding = nn.Linear(self.patch_dim, dim)

        # position encoding
        self.learn_pos_encoding = learn_pos_encoding
        fourier_dim = dim // 6 # sin/cos features for each dim
        w = torch.arange(fourier_dim) / (fourier_dim - 1)
        w = (1. / (10_000 ** w)).repeat(3)
        self.pos_encoding_freqs = nn.Parameter(
            w.log() if self.learn_pos_encoding else w, requires_grad=self.learn_pos_encoding
        )
        self.init_pos_grid(axis_sizes)

        # transformer stack
        self.blocks = nn.ModuleList([
            Block(
                dim, num_heads, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop,
                checkpoint_grads=checkpoint_grads, attn_drop=attn_drop,
                proj_drop=proj_drop
            ) for _ in range(depth)
        ])

        # norm layer
        self.out_norm = nn.LayerNorm(dim, eps=1e-6)

        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(dim))

        self.head = MLP2(
            units=[hidden_dim, hidden_dim, 6],
            act='relu',
            out_act='sigmoid',
            drop=0.
        )

    def init_pos_grid(self, axis_sizes):
        self.num_patches = [s // p for s, p in zip(axis_sizes, self.patch_shape)]
        for i, n in enumerate(self.num_patches): # axis values for each dim
            self.register_buffer(f'grid_{i}', torch.arange(n)*(2*math.pi/n))

    def pos_encoding(self): # TODO: Simplify for fixed dim=3
        grids = [getattr(self, f'grid_{i}') for i in range(3)]
        coords = torch.meshgrid(*grids, indexing='ij')

        if self.learn_pos_encoding:
            freqs = self.pos_encoding_freqs.exp().chunk(3)
        else:
            freqs = self.pos_encoding_freqs.chunk(3)

        features = [
            trig_fn(x.flatten()[:,None] * w[None, :])
            for (x, w) in zip(coords, freqs) for trig_fn in (torch.sin, torch.cos)
        ]
        return torch.cat(features, dim=1)

    def forward(self, x, mask=None):
        """
        Forward pass of ViT.
        :param x   : tensor of spatial inputs with shape (batch_size, channels, *axis_sizes)
        :param mask: a tensor of patch indices that should be masked out of `x`.
        """

        # patchify input
        # x -> (batch_size, number_of_patches, voxels_per_patch)
        x = self.to_patches(x)
        
        # embed
        # x -> (batch_size, number_of_patches, embedding_dim)
        x = self.embedding(x)

        # apply mask and position encoding
        if self.use_mask_token:
            if mask is not None:
                x = self.apply_mask_tokens(x, mask)
            x = x + self.pos_encoding()
        
        # process patches with transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)

        return x

    def to_patches(self, x):
        x = rearrange(
            x, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)',
            **dict(zip(('p1', 'p2', 'p3'), self.patch_shape))
        )
        return x

    def apply_mask_tokens(self, x, mask_idcs):
        """
        Replaces patch embeddings in `x` with the network's mask token at indices speficied by `mask`.

        :param x   : input tensor with shape (B [batch size], T [number of patches], D [embed dim])
        :param mask: tensor with shape (B, T) containing indices in the range [0,T)
        """
        B, T = x.shape[:2]
        full_mask_token = repeat(self.mask_token, 'd -> b t d', b=B, t=T)
        # construct boolean mask
        mask = torch.zeros((B, T), device=x.device).scatter_(-1, mask_idcs, 1).bool()
        return torch.where(mask[..., None], full_mask_token, x)          

class Block(nn.Module):
    def __init__(
            self, hidden_size, num_heads, mlp_ratio=4.0, mlp_drop=0., checkpoint_grads=False,
            **attn_kwargs
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **attn_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu,
            drop=mlp_drop
        )
        self.checkpoint_grads = checkpoint_grads

    def forward(self, x):
        if self.checkpoint_grads:
            x = x + checkpoint(self.attn, self.norm1(x), use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
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
    
class MLP2(nn.Module):

    def __init__(
        self,
        units,
        act,
        out_act,
        drop
    ):

        # units, act, drop=None

        super(MLP2, self).__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(a, b) for a, b in pairwise(units)])
        self.act = getattr(F, act)
        self.out_act = getattr(F, out_act) if out_act else None
        self.drop = nn.Dropout(drop) if drop else None

    def forward(self, x):
        
        for linear in self.linear_layers[:-1]:
            
            x = linear(x)
            x = self.act(x)
            if self.drop is not None:
                x = self.drop(x)
        
        x = self.linear_layers[-1](x)
        if self.out_act is not None:
            x = self.out_act(x)        
        
        return x
    
class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PretrainedViT(ViT):
    """
    A class for initializing pretrained ViTs.
    """

    def __init__(
        self,
        backbone_dir = "runs/pretraining_micro/huge_775",
        drop_head = True,
        frozen = True,
        # add_head = False,
        # head_args = {
        #     '_target_': 'src.networks.MLP',
        #     'cfg': {
        #         'units': [
        #           144,
        #           144,
        #           6],
        #         'act': 'relu',
        #         'out_act': 'sigmoid',
        #         'drop': 0.}
        # },
        # adapt_res = False,
        # adapt_args = {
        #       'channels': 4,
        #       'downsample_factor': 5,
        #       'extra_proj': True,
        #       'replace_embedding': False
        # },
        # use_input_conv = False,
        # input_conv_args = {
        #     'channels': 8,
        #     'kernel1': [4,4,5],
        #     'stride1': [2,2,3],
        #     'kernel2': [3, 3, 4],
        #     'stride2': [2, 2, 3],
        #     'conv_out_dim': 640
        # },
        # interp_pos_encoding = False
    ):

        # read backbone config
        bb_dir = backbone_dir
        # bcfg = get_prev_config(bb_dir)

        # load backbone state
        model_state = torch.load(os.path.join(bb_dir, 'model.pt'))["model"]
        net_state = {
            k.replace('net.', ''): v for k,v in model_state.items() if k.startswith('net.')
        }
        
        # initialize network and load weights
        super().__init__()
        self.load_state_dict(net_state)
        
        # delete the head module used in pretraining
        if drop_head and hasattr(self, 'head'):
            del self.head

        # freeze weights and set to eval mode
        if frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            
        # init new head or input adaption if needed
        # if add_head:
        #     self.head = instantiate(cfg.head)
        # if adapt_res:
        #     init_adaptor(cfg.adaptor)
        # if use_input_conv:
        #     init_input_conv(cfg.input_conv)            
        # if interp_pos_encoding:
        #     bb.init_pos_grid(cfg.data_shape)              

class Phi3ImageEmbeddingSkatr(nn.Module):
    """Phi3 Skatr Image embedding."""

    def __init__(self, config: PretrainedConfig, wte=None, pretrained_backbone_dir=None, **kwargs) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        self.wte = wte

        if isinstance(config.img_processor, dict) and config.img_processor.get('name', None) == 'vit':
            self.img_processor = PretrainedViT(backbone_dir=pretrained_backbone_dir)
            image_dim_out = config.img_processor['image_dim_out']
            self.num_img_tokens = config.img_processor['num_img_tokens']
            self.patch_sizes = config.img_processor['patch_sizes']
        else:
            raise NotImplementedError(f'img_processor = {config.img_processor}, not implemented')

        self.image_dim_out = image_dim_out
        self.img_sizes = None

        # global_gn and sub_gn for hd transform, serves as line separator
        # with_hd_transform and with_learnable_separator should have same value
        # self.use_hd_transform = kwargs.get('use_hd_transform', False)
        # self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        # projection_cls = kwargs.get('projection_cls', 'linear')
        # if projection_cls == 'linear':
        #     self.img_projection = nn.Linear(image_dim_out, hidden_size)
        # elif projection_cls == 'mlp':
        #     dim_projection = hidden_size
        #     depth = 2
        #     layers = [nn.Linear(image_dim_out, dim_projection)]
        #     for _ in range(1, depth):
        #         layers.extend([nn.GELU(),
        #                         nn.Linear(dim_projection, dim_projection)])
        #     self.img_projection = nn.Sequential(*layers)
        # else:
        #     raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.img_projection = nn.Linear(image_dim_out, hidden_size)

        self.vocab_size = config.vocab_size
        self.img_features = None

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'


    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        # LAYER_IDX = self.layer_idx
        # TYPE_FEATURE = self.type_feature

        # img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
        # img_feature = img_processor_output.hidden_states[LAYER_IDX]

        img_feature = self.img_processor(img_embeds)
        return img_feature

        # if TYPE_FEATURE == "patch":
        #     patch_feature = img_feature[:, 1:]
        #     return patch_feature

        raise NotImplementedError

    def forward(
        self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, image_sizes=None
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # positions for image tokens
        positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=True)
        has_image = len(positions[0].tolist()) > 0
        input_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size).detach()
        hidden_states = self.wte(input_ids)

        if has_image:
            num_images, num_crops, c, h, w, d = pixel_values.shape
            assert c == 1 and h == self.patch_sizes[0] and w == self.patch_sizes[1] and d == self.patch_sizes[2]
            img_features = self.get_img_features(pixel_values.flatten(0, 1)).reshape(
                num_images, num_crops, -1, self.image_dim_out
            )
            # image_features_proj = self.hd_feature_transform(img_features, image_sizes)
            image_features_proj = self.img_projection(img_features)
            hidden_states = hidden_states.index_put(
                positions, image_features_proj, accumulate=False
            )

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states

    # def hd_feature_transform(self, image_features, image_sizes):
    #     """
    #     image_features: (num_images, num_crops+1, 24*24, 1024)
    #     """
    #     assert (
    #         self.hd_transform_order == 'sub_glb'
    #     ), f'hd_transform_order `{self.hd_transform_order}` not implemented'
    #     if isinstance(self.img_projection, nn.Sequential):
    #         target_device = self.img_projection[0].bias.device
    #         target_dtype = self.img_projection[0].bias.dtype
    #     else:  # It's a single nn.Linear layer
    #         target_device = self.img_projection.bias.device
    #         target_dtype = self.img_projection.bias.dtype

    #     global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
    #     # global feature can be viewed as a special HD case with num_crops 1x1
    #     global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
    #     global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

    #     all_image_embeddings = []
    #     # need a for loop to process each image because of different image sizes
    #     # (patch arrangement is different for each image)
    #     for i, img_size in enumerate(image_sizes):
    #         h, w = img_size
    #         h_crop = h // 336
    #         w_crop = w // 336
    #         num_crops = h_crop * w_crop

    #         # NOTE: real num_crops is padded
    #         # (num_crops, 24*24, 1024)
    #         sub_image_features = image_features[i, 1 : 1 + num_crops]
    #         sub_image_features_hd = self.reshape_hd_patches_2x2merge(
    #             sub_image_features, h_crop, w_crop
    #         )
    #         sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

    #         # [sub features, separator, global features]
    #         all_image_embeddings.extend(
    #             [
    #                 sub_image_features_hd_newline.squeeze(0),  # (h_crop*12*(w_crop*12+1), 4096)
    #                 self.glb_GN.squeeze(0),
    #                 global_image_features_hd_newline[i],
    #             ]
    #         )

    #     image_features_proj = self.img_projection(
    #         torch.cat(all_image_embeddings, dim=0).to(target_device).to(target_dtype)
    #     )

    #     return image_features_proj

    # def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
    #     """
    #     image_features: (num_images*num_crops, 24*24, 1024)
    #     output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
    #     """
    #     N, L, C = image_features.shape
    #     assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
    #     num_images = N // (h_crop * w_crop)
    #     H = int(L**0.5)
    #     image_features_hd = (
    #         image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
    #         .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
    #         .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
    #         .reshape(N, -1, 4 * C)  # N, 144, 4096
    #         .reshape(
    #             num_images, h_crop, w_crop, H // 2, H // 2, -1
    #         )  # n_img, h_crop, w_crop, 12, 12, 4096
    #         .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
    #         .reshape(
    #             num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
    #         )  # n_img, h_crop*12, w_crop*12, 4096
    #     )

    #     # alternative implementation using einops
    #     # from einops import rearrange
    #     # image_features_nhwc = rearrange(
    #     #     image_features,
    #     #     'N (H W) c -> N H W c',
    #     #     H=H,
    #     #     W=H,
    #     # )
    #     # image_features_2x2merge = rearrange(
    #     #     image_features_nhwc,
    #     #     'N (h h_pool) (w w_pool) c -> N h w (h_pool w_pool c)',
    #     #     h_pool=2,
    #     #     w_pool=2,
    #     # )
    #     # image_features_hd = rearrange(
    #     #     image_features_2x2merge,
    #     #     '(n_img h_crop w_crop) h w C -> n_img (h_crop h) (w_crop w) C',
    #     #     h_crop=h_crop,
    #     #     w_crop=w_crop,
    #     # )

    #     return image_features_hd

    # def add_image_newline(self, image_features_hd):
    #     """
    #     image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
    #     output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
    #     """
    #     num_images, h, w, hid_dim = image_features_hd.shape
    #     # add the newline token to the HD image feature patches
    #     newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
    #     image_features_hd_newline = torch.cat(
    #         [image_features_hd, newline_embeddings], dim=2
    #     ).reshape(num_images, -1, hid_dim)
    #     return image_features_hd_newline
    