"""Load the W1 world model (LeWM recipe, top + wrist RGB -> one 192-d latent) for the paper-vs-hardware W1 posts.

    from w1_load import load_w1
    model = load_w1()                         # downloads ravediamond/lewm-w2-so101-bowls @ d0336db6, w1/weights_final.pt
    model = load_w1("weights/w1/weights_final.pt", device="cuda")

Strict load only: W1 was saved with transformers 5.19.0 (this image's /opt/pydeps-w1). Any missing or unexpected key
raises; there is no key remap and no strict=False, because a silently dropped key leaves random encoder weights (the
voided-gates post). Same construction as the research rig's W1LiveMonitor.from_checkpoint.
"""
from pathlib import Path

import torch
from torch import nn

REPO, REVISION, FILE = "ravediamond/lewm-w2-so101-bowls", "d0336db6", "w1/weights_final.pt"


def checkpoint_path(path=None):
    if path:
        return Path(path)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, FILE, revision=REVISION))


def load_w1(path=None, device=None, dtype=None):
    import hydra
    from omegaconf import OmegaConf
    import transformers
    if transformers.__version__ != "5.19.0":
        raise RuntimeError(f"W1 loads strictly only with transformers 5.19.0, found {transformers.__version__}")
    ckpt = torch.load(checkpoint_path(path), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = hydra.utils.instantiate(cfg.model)
    model.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    if cfg.loss.idm.weight > 0:
        h = cfg.loss.idm.hidden_dim
        model.idm = nn.Sequential(nn.Linear(2 * cfg.embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                  nn.Linear(h, cfg.data.action_dim))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if str(dev).startswith("cuda") else torch.float32)
    return model.to(dev, dtype).eval()
