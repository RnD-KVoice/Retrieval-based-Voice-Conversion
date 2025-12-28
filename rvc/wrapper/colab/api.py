import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from scipy.io import wavfile

from rvc.modules.vc.modules import VC
from rvc.wrapper.colab.setup import setup_colab_environment, check_gpu
from rvc.wrapper.colab import storage

logger = logging.getLogger(__name__)

_vc_state = {"instance": None, "current_model": None}


def create_app(storage_path="/content/models", assets_path="/content/assets", auto_setup=True):
    app = FastAPI(
        title="RVC Colab API",
        description="Voice conversion API for Google Colab",
        version="1.0.0"
    )

    if auto_setup:
        logger.info("Running automatic environment setup...")
        setup_colab_environment(storage_path, assets_path)

    @app.get("/")
    async def root():
        return {
            "name": "RVC Colab API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "inference": "/api/v1/inference",
                "models": "/api/v1/models",
                "upload": "/api/v1/models/upload",
                "docs": "/docs"
            }
        }

    @app.get("/health")
    async def health():
        gpu_info = check_gpu()
        storage_info = storage.get_storage_info(storage_path)

        return {
            "status": "healthy",
            "gpu_available": gpu_info["available"],
            "gpu_name": gpu_info["device_name"],
            "gpu_memory_gb": gpu_info["memory_total_gb"],
            "loaded_model": _vc_state["current_model"],
            "models_count": storage_info["models_count"],
            "available_models": [m["filename"] for m in storage.list_models(storage_path)]
        }

    @app.get("/api/v1/models")
    async def list_models():
        models = storage.list_models(storage_path)
        return {"count": len(models), "models": models}

    @app.post("/api/v1/models/upload")
    async def upload_model(
        model_file: UploadFile = File(...),
        index_file: UploadFile = File(None)
    ):
        try:
            if not model_file.filename.endswith('.pth'):
                raise HTTPException(status_code=400, detail="Model file must be .pth")

            model_data = await model_file.read()
            model_info = storage.save_model(model_data, model_file.filename, storage_path)

            index_info = None
            if index_file:
                if not index_file.filename.endswith('.index'):
                    logger.warning(f"Index file should have .index extension")

                index_data = await index_file.read()
                index_info = storage.save_index(index_data, index_file.filename, storage_path)

            return {
                "success": True,
                "model": model_info,
                "index": index_info,
                "message": f"Model '{model_file.filename}' uploaded successfully"
            }

        except Exception as e:
            logger.error(f"Model upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/inference")
    async def inference(
        audio_file: UploadFile = File(...),
        model_name: str = Form(...),
        f0_up_key: int = Form(0, ge=-12, le=12),
        f0_method: str = Form("rmvpe", regex="^(pm|harvest|crepe|rmvpe)$"),
        index_rate: float = Form(0.75, ge=0.0, le=1.0),
        filter_radius: int = Form(3, ge=0, le=10),
        resample_sr: int = Form(0, ge=0),
        rms_mix_rate: float = Form(0.25, ge=0.0, le=1.0),
        protect: float = Form(0.33, ge=0.0, le=0.5)
    ):
        temp_input_path = None

        try:
            audio_data = await audio_file.read()
            file_size_mb = len(audio_data) / (1024 * 1024)

            if file_size_mb > 50:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large ({file_size_mb:.1f} MB). Max 50 MB."
                )

            logger.info(f"Processing: model={model_name}, pitch={f0_up_key}, method={f0_method}")

            load_model(model_name, storage_path)

            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix)
            temp_input_path = temp_input.name
            temp_input.write(audio_data)
            temp_input.close()

            model_stem = model_name.replace('.pth', '')
            index_file = storage.find_index(model_stem, storage_path)

            hubert_path = os.getenv("hubert_path")
            if not hubert_path or not Path(hubert_path).exists():
                raise HTTPException(status_code=500, detail="HuBERT model not found")

            tgt_sr, audio_opt, times, error_info = _vc_state["instance"].vc_inference(
                sid=0,
                input_audio_path=temp_input_path,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                f0_file=None,
                index_file=index_file,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                hubert_path=hubert_path
            )

            if error_info is not None:
                logger.error(f"Inference error: {error_info}")
                raise HTTPException(status_code=500, detail=f"Inference failed: {error_info}")

            wav_io = BytesIO()
            wavfile.write(wav_io, tgt_sr, audio_opt)
            wav_io.seek(0)

            logger.info(f"Completed. Times: {times}")

            return StreamingResponse(
                wav_io,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=converted_{audio_file.filename}.wav",
                    "X-Processing-Time-NPY": str(times["npy"]),
                    "X-Processing-Time-F0": str(times["f0"]),
                    "X-Processing-Time-Infer": str(times["infer"])
                }
            )

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if temp_input_path and Path(temp_input_path).exists():
                try:
                    Path(temp_input_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

    @app.delete("/api/v1/models/{model_name}")
    async def delete_model_endpoint(model_name: str):
        try:
            if _vc_state["current_model"] == model_name:
                unload_model()

            success = storage.delete_model(model_name, storage_path)

            if not success:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            return {"success": True, "message": f"Model '{model_name}' deleted"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def load_model(model_name, base_path="/content/models"):
    try:
        model_path = storage.get_model_path(model_name, base_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if _vc_state["current_model"] == model_name and _vc_state["instance"] is not None:
        logger.info(f"Model '{model_name}' already loaded")
        return

    if _vc_state["instance"] is not None:
        unload_model()

    logger.info(f"Loading model: {model_name}")
    try:
        _vc_state["instance"] = VC()
        n_spk, protect_values, index_path = _vc_state["instance"].get_vc(model_path)

        _vc_state["current_model"] = model_name

        logger.info(f"Model loaded: {model_name}")
        logger.info(f"  Speakers: {n_spk}, Index: {index_path or 'None'}")

    except Exception as e:
        _vc_state["instance"] = None
        _vc_state["current_model"] = None
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def unload_model():
    if _vc_state["instance"] is not None:
        logger.info(f"Unloading model: {_vc_state['current_model']}")

        if hasattr(_vc_state["instance"], 'net_g') and _vc_state["instance"].net_g is not None:
            del _vc_state["instance"].net_g

        if hasattr(_vc_state["instance"], 'hubert_model'):
            del _vc_state["instance"].hubert_model

        del _vc_state["instance"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _vc_state["instance"] = None
        _vc_state["current_model"] = None

        logger.info("Model unloaded")
