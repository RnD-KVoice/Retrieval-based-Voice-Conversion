import os
import torch

from fairseq import checkpoint_utils

_original_torch_load = torch.load

def _torch_load_wrapper(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_wrapper


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
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    return hubert_model.eval()
