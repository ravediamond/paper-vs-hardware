"""Path A: upstream TurboVLA's own trainer (turbovla/training/trainer.py, unchanged) on the SO-101 bowls data.

Two substitutions, nothing else: its LIBERO RLDS dataset -> So101BowlsDataset (same sample format), and the model is
initialised from the full released LIBERO checkpoint (load_libero_init) instead of Grounding DINO's interaction
layers. Every other argument goes to upstream's parser unchanged.

    python train_so101.py --upstream third_party/TurboVLA --libero-ckpt <turbovla_libero.pth> --stats stats.json \\
        -- --dataset_dir <train cache (so101_data.build_cache)> --dinov3_path <dir> --bert_path <dir> --checkpoint_dir <out> ...
"""

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path

# the action/state width differs from LIBERO's 7 / 8: these layers keep their fresh init (FAITHFUL_PROTOCOL.md)
REINIT = ("action_head.state_projection.net.0.", "action_head.state_projection.net.1.",
          "action_head.decoder.action_projection.layers.2.")
# the LIBERO checkpoint has 12 action queries: at another chunk length they keep their fresh init too (H50_BRIEF step 4)
QUERIES = "action_head.decoder.action_queries."

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", type=Path, required=True)
ap.add_argument("--libero-ckpt", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--init-report", default="init_report.json")
args, rest = ap.parse_known_args()
rest = [a for a in rest if a != "--"]
chunk = int(rest[rest.index("--chunk_size") + 1]) if "--chunk_size" in rest else 12
reinit = REINIT + ((QUERIES,) if chunk != 12 else ())
sys.path.insert(0, str(args.upstream))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# libero_rlds imports TensorFlow at module level for the RLDS reader this script replaces; vla_collate_fn (the part
# used) is plain torch. Empty stand-ins let the import pass without installing TensorFlow. The libraries that probe
# for TensorFlow (transformers) are imported first, so they still see that it is absent.
import transformers  # noqa: E402,F401

for name in ("tensorflow", "tensorflow_datasets"):
    stub = types.ModuleType(name)
    stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules.setdefault(name, stub)

import turbovla.training.trainer as T  # noqa: E402
from so101_data import So101BowlsDataset, load_libero_init  # noqa: E402

So101BowlsDataset.stats_path = args.stats
T.LiberoRLDSDataset = So101BowlsDataset
_build = T.build_model_architecture


def build_with_libero_init(targs):
    model = _build(targs)
    report = load_libero_init(model, args.libero_ckpt, reinit)
    report["checkpoint"] = args.libero_ckpt
    Path(args.init_report).write_text(json.dumps(report, indent=1))
    print(f"LIBERO init: {report['loaded']} tensors loaded, re-initialised {report['reinitialised']}", flush=True)
    return model


T.build_model_architecture = build_with_libero_init
sys.argv = [sys.argv[0], *rest]
T.train_model()
