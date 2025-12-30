import os
import logging
from pathlib import Path
import torch

from rvc.modules.vc.modules import VC
from rvc.configs.config import Config

logger = logging.getLogger(__name__)

_rvc_state = {
    "vc": None,
    "current_model": None,
    "hubert_model": None
}


def load_model(model_path, index_path=None):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if _rvc_state["current_model"] == model_path and _rvc_state["vc"] is not None:
        logger.info(f"Model already loaded: {model_path}")
        return

    logger.info(f"Loading RVC model: {model_path}")

    _rvc_state["vc"] = VC()
    n_spk, protect_values, detected_index = _rvc_state["vc"].get_vc(model_path)

    _rvc_state["current_model"] = model_path

    logger.info(f"Model loaded - Speakers: {n_spk}, Index: {detected_index or index_path or 'None'}")


def convert_voice(
    source_path,
    model_path,
    output_path,
    f0_up_key=0,
    f0_method="rmvpe",
    index_path=None,
    index_rate=0.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33
):
    if not Path(source_path).exists():
        raise FileNotFoundError(f"Source audio not found: {source_path}")

    load_model(model_path, index_path)

    hubert_path = os.getenv("hubert_path")
    if not hubert_path or not Path(hubert_path).exists():
        raise RuntimeError("HuBERT model not found. Set 'hubert_path' environment variable.")

    logger.info(f"Converting: {source_path} with model {model_path}")
    logger.info(f"Parameters: pitch={f0_up_key}, method={f0_method}, index_rate={index_rate}")

    tgt_sr, audio_opt, times, error_info = _rvc_state["vc"].vc_inference(
        sid=0,
        input_audio_path=str(source_path),
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        f0_file=None,
        index_file=str(index_path) if index_path else None,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        hubert_path=hubert_path
    )

    if error_info is not None:
        raise RuntimeError(f"Voice conversion failed: {error_info}")

    import soundfile as sf
    sf.write(output_path, audio_opt, tgt_sr)

    logger.info(f"Conversion complete: {output_path}")
    logger.info(f"Processing times - NPY: {times['npy']}s, F0: {times['f0']}s, Infer: {times['infer']}s")

    return output_path


def unload_model():
    if _rvc_state["vc"] is not None:
        logger.info(f"Unloading RVC model: {_rvc_state['current_model']}")

        if hasattr(_rvc_state["vc"], 'net_g'):
            del _rvc_state["vc"].net_g
        if hasattr(_rvc_state["vc"], 'hubert_model'):
            del _rvc_state["vc"].hubert_model

        del _rvc_state["vc"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _rvc_state["vc"] = None
        _rvc_state["current_model"] = None

        logger.info("Model unloaded")
