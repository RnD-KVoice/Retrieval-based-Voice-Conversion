import os
from functools import partial

import torch

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if str(sid).split(".")[0] in f
        ),
        "",
    )


def load_hubert(config, hubert_path: str):
    # PyTorch 2.6+ defaults to weights_only=True which breaks fairseq's
    # internal torch.load calls (UnpicklingError).  Monkey-patch torch.load
    # to force weights_only=False while loading the HuBERT checkpoint.
    _original_torch_load = torch.load
    torch.load = partial(_original_torch_load, weights_only=False)
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path],
            suffix="",
        )
    finally:
        torch.load = _original_torch_load

    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    return hubert_model.eval()
