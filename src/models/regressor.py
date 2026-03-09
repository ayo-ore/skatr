import torch
import torch.nn.functional as F

from src import networks
from src.models.base_model import Model


class Regressor(Model):

    def __init__(self, net, loss="l1", summary_net=None, summarize=False):

        super().__init__(net, summary_net)

        self.summarize = summarize

        match loss:
            case "l1":
                self.loss = F.l1_loss
            case "l2":
                self.loss = F.mse_loss
            case _:
                raise ValueError(f"Unknown loss {self.loss}")

    def batch_loss(self, batch):
        preds = self.forward(batch)
        return self.loss(batch.labels, preds)

    def forward(self, batch):

        # if hasattr(self, "summary_net") and not self.cfg.data.summarize:
        if self.summarize:

            x = batch.images
            x = self.summary_net(x)
            # if not hasattr(self.bb, 'head') and self.net.cfg.arch == 'MLP': # weird...
            if (
                not hasattr(self.summary_net, "head") and self.cfg.net.arch == "MLP"
            ):  # TODO: Clean
                x = x.mean(1)  # (B, T, D) --> (B, D)

        elif hasattr(self, "summary_net"):
            x = batch.summaries
        else:
            x = batch.images

        return self.net(x)

    @torch.inference_mode()
    def predict(self, x):
        return self.forward(x)


class GaussianRegressor(Regressor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.const_sigma_frac = cfg.const_sigma_frac
        self.stop_sigma = int(bool(self.const_sigma_frac))

    def batch_loss(self, batch):

        mu, sigma = self(batch.images)
        # optionally fix sigma constant
        sigma = (1 - self.stop_sigma) * sigma + self.stop_sigma
        # gaussian likelihood
        loss = 0.5 * ((batch.labels - mu) / sigma) ** 2 + sigma.log()

        return loss.mean()

    def forward(self, x):
        logit_mu, invsp_sig = super().forward(x).tensor_split(2, dim=-1)
        mu = F.sigmoid(logit_mu)
        sigma = F.softplus(invsp_sig)
        return mu, sigma

    @torch.inference_mode()
    def predict(self, x):
        return self.forward(x)

    def update(self, loss, optimizer, scaler, step=None, total_steps=None):

        # default update
        super().update(loss, optimizer, scaler)

        # enable variance
        if step / total_steps >= self.const_sigma_frac:
            self.stop_sigma = 0.0
