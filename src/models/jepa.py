import copy
import torch
import torch.nn.functional as F

from src.models.base_model import Model
from src.utils import augmentations, masks


class JEPA(Model):

    def __init__(
        self,
        net,
        predictor,
        sim="l1",
        ema_momentum=0.9997,
        momentum_schedule=True,
        init_tgt_as_ctx=True,
        summary_net=None,
    ):
        super().__init__(net=net, summary_net=summary_net)
        self.predictor = predictor
        self.ema_momentum = ema_momentum
        self.momentum_schedule = momentum_schedule
        self.ctx_encoder = self.net
        self.tgt_encoder = copy.deepcopy(self.ctx_encoder)
        if not init_tgt_as_ctx:
            # re-initialize weights
            for module in self.tgt_encoder.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        self.augment = augmentations.RotateAndReflect()

        match sim:
            case "l2":
                self.sim = lambda x1, x2: -F.mse_loss(x1, x2)
            case "l1":
                self.sim = lambda x1, x2: -F.l1_loss(x1, x2)
            case "smooth_l1":
                self.sim = lambda x1, x2: -F.smooth_l1_loss(x1, x2)

    def batch_loss(self, batch):

        images, tgt_masks, ctx_masks = batch

        # get target token embeddings
        with torch.no_grad():
            tgt_tokens = self.tgt_encoder(images)

        loss = 0.0
        for ctx_mask, tgt_mask in zip(ctx_masks, tgt_masks):
            # WARNING: Assumes each target mask has it's own context. Repeat ctx_mask otherwise

            # get context token embeddings and predict
            ctx_tokens = self.ctx_encoder(images, mask=ctx_mask)
            prd_tokens = self.predictor(ctx_tokens, ctx_mask, tgt_mask)

            # keep only target tokens in current block
            local_tgt_tokens = masks.gather_tokens(tgt_tokens, tgt_mask)

            # similarity loss
            loss += -self.sim(prd_tokens, local_tgt_tokens)

        return loss

    def update(
        self, loss, optimizer, scaler, step=None, total_steps=None, gradient_norm=None
    ):

        # student update
        super().update(loss, optimizer, scaler, gradient_norm=gradient_norm)

        # teacher update via exponential moving average of student
        tau = self.ema_momentum
        if self.momentum_schedule:  # linear increase to tau=1
            frac = step / total_steps
            tau = tau + (1 - tau) * frac

        for ps, pt in zip(self.ctx_encoder.parameters(), self.tgt_encoder.parameters()):
            pt = tau * pt + (1 - tau) * ps

    def forward(self, x, mask=False):
        return self.ctx_encoder(x, mask=mask)

    @torch.inference_mode()
    def embed(self, x):
        return self.ctx_encoder(x)
