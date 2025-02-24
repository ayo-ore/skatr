import copy
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.base_model import Model
from src.networks import NdPredictorViT
from src.utils import augmentations, masks


class JEPA(Model):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.ctx_encoder = self.net
        self.tgt_encoder = (
            copy.deepcopy(self.ctx_encoder)
            if cfg.init_tgt_as_ctx
            else self.net.__class__(cfg.net)
        )
        self.predictor = NdPredictorViT(cfg.predictor)
        
        self.augment = augmentations.RotateAndReflect()

        match cfg.sim:
            case "l2":
                self.sim = lambda x1, x2: -F.mse_loss(x1, x2)
            case "l1":
                self.sim = lambda x1, x2: -F.l1_loss(x1, x2)
            case "smooth_l1":
                self.sim = lambda x1, x2: -F.smooth_l1_loss(x1, x2)

    def batch_loss(self, batch):

        # augment batch
        x1 = batch[0]
        x2 = self.augment(x1) if self.cfg.augment else x1

        # sample masks
        num_patches = self.ctx_encoder.num_patches
        tgt_masks, ctx_masks = masks.JEPA_mask(
            num_patches, self.cfg.masking, batch_size=x2.size(0), device=x2.device
        )

        # get target token embeddings
        with torch.no_grad():
            tgt_tokens = self.tgt_encoder(x1)

        loss = 0.0
        for ctx_mask, tgt_mask in zip(ctx_masks, tgt_masks):
            # WARNING: Assumes each target mask has it's own context. Repeat ctx_mask otherwise

            # get context token embeddings and predict
            ctx_tokens = self.ctx_encoder(x2, mask=ctx_mask)
            prd_tokens = self.predictor(ctx_tokens, ctx_mask, tgt_mask)

            # keep only target tokens in current block
            local_tgt_tokens = masks.gather_tokens(tgt_tokens, tgt_mask)

            # similarity loss
            loss += -self.sim(prd_tokens, local_tgt_tokens)

        return loss

    def update(self, loss, optimizer, scaler, step=None, total_steps=None):

        # student update
        super().update(loss, optimizer, scaler)

        # teacher update via exponential moving average of student
        tau = self.cfg.ema_momentum
        if self.cfg.momentum_schedule:  # linear increase to tau=1
            frac = step / total_steps
            tau = tau + (1 - tau) * frac

        for ps, pt in zip(self.ctx_encoder.parameters(), self.tgt_encoder.parameters()):
            pt = tau * pt + (1 - tau) * ps

    def forward(self, x, mask=False):
        return self.ctx_encoder(x, mask=mask)

    @torch.inference_mode()
    def embed(self, x):
        return self.ctx_encoder(x)
