"""W3: spatial LeWM. DINOv2 ViT-S/14 (initialized from the pretrained weights, trained end-to-end) -> patch tokens pooled to a
grid x grid map per view -> per-token BatchNorm projector (SIGReg is applied to these tokens) -> a latent of views x grid^2 tokens
of width 384 per frame. The predictor is the W2 one (train_w2.W2Predictor: AdaLN-zero blocks over the tokens of a C = 3 frame window,
residual head), so W2 and W3 differ only in the encoder (frozen + precomputed vs trained end-to-end with SIGReg).

Plugs into train_lewm_mv.py through `spatial:` in the config (config/lewm_mv_spatial_so101.yaml): the script calls
`encode_views` instead of its CLS path, and `predict(window latents, window actions)` as for the vector model.
"""

import torch
import torch.nn.functional as F
from torch import nn

import train_w2 as W


class SpatialLeWM(nn.Module):
    def __init__(self, views: int = 2, grid: int = 4, dino: str = "facebook/dinov2-small", grad_checkpoint: bool = True,
                 proj_hidden: int = 2048, freeze_blocks: int = 0, **predictor_kw):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(dino)
        if grad_checkpoint:     # non-reentrant: with frozen early blocks the reentrant variant would drop the gradients of the trainable ones
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if freeze_blocks:       # freeze the patch embedding and the first `freeze_blocks` transformer blocks (deviation recorded in G1_PROTOCOL.md)
            for prm in self.encoder.embeddings.parameters():
                prm.requires_grad = False
            for blk in self.encoder.encoder.layer[:freeze_blocks]:
                for prm in blk.parameters():
                    prm.requires_grad = False
        self.grid, self.views, self.width = grid, views, self.encoder.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(self.width, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.GELU(),
                                       nn.Linear(proj_hidden, self.width))
        self.predictor = W.W2Predictor(tokens=views * grid * grid, feat=self.width, d=self.width, **predictor_kw)
        self.action_encoder = nn.Identity()                       # the predictor embeds the normalized actions itself

    def encode_views(self, batch, views):
        """batch[view]: (B, T, 3, H, W) -> z (B, T, views * grid^2, 384), actions (B, T, 6) unchanged."""
        x = torch.stack([batch[v] for v in views], dim=2)         # (B, T, V, 3, H, W)
        for sub in (self.encoder, self.projector, self.predictor):
            if sub.training != self.training:   # the trainer leaves the encoder in eval mode (HF from_pretrained): that silently disables checkpointing
                sub.train(self.training)
        if self.training and not getattr(self, "_dbg", False):    # one-time diagnostic, first training call
            self._dbg = True
            l0 = self.encoder.encoder.layer[-1]
            print(f"[spatial] model.training={self.training} encoder.training={self.encoder.training} last_layer.training={l0.training} "
                  f"projector.training={self.projector.training} predictor.training={self.predictor.training} layer.gradient_checkpointing={getattr(l0, 'gradient_checkpointing', None)} grad_enabled={torch.is_grad_enabled()} "
                  f"autocast={torch.is_autocast_enabled()} images={x.shape[0] * x.shape[1] * x.shape[2]}", flush=True)
        b, t, v = x.shape[:3]
        h = self.encoder(pixel_values=x.flatten(0, 2).to(next(self.encoder.parameters()).dtype)).last_hidden_state[:, 1:]
        n, p, c = h.shape
        s = int(p ** 0.5)
        h = F.adaptive_avg_pool2d(h.reshape(n, s, s, c).permute(0, 3, 1, 2), self.grid).flatten(2).transpose(1, 2)  # (n, g^2, c)
        z = self.projector(h.reshape(-1, c)).view(b, t, v * self.grid ** 2, c)
        return z, batch["action"]

    def predict(self, emb, act):
        return self.predictor(emb, act)
