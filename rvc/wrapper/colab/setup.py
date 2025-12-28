import os
import logging
from pathlib import Path
import requests
import torch

logger = logging.getLogger(__name__)

HUBERT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"


def download_file(url, destination, desc="Downloading"):
    try:
        if destination.exists():
            logger.info(f"{destination.name} already exists, skipping")
            return True

        logger.info(f"Downloading {desc} from {url}")

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        destination.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        chunk_size = 8192

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if downloaded % (5 * 1024 * 1024) < chunk_size:
                        progress_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        logger.info(f"  Progress: {progress_mb:.1f} / {total_mb:.1f} MB")

        file_size_mb = destination.stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded {destination.name} ({file_size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if destination.exists():
            destination.unlink()
        return False


def download_base_models(assets_dir="/content/assets"):
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)

    logger.info("Checking for required base models...")

    results = {}

    hubert_path = assets_path / "hubert_base.pt"
    results["hubert"] = download_file(HUBERT_URL, hubert_path, "HuBERT model")

    rmvpe_path = assets_path / "rmvpe.pt"
    results["rmvpe"] = download_file(RMVPE_URL, rmvpe_path, "RMVPE model")

    if results["hubert"]:
        os.environ["hubert_path"] = str(hubert_path)
        logger.info(f"Set hubert_path: {hubert_path}")

    if results["rmvpe"]:
        os.environ["rmvpe_root"] = str(assets_path)
        logger.info(f"Set rmvpe_root: {assets_path}")

    return results


def check_gpu():
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": None,
        "memory_total_gb": 0,
        "cuda_version": None
    }

    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["cuda_version"] = torch.version.cuda

        props = torch.cuda.get_device_properties(0)
        gpu_info["memory_total_gb"] = round(props.total_memory / (1024**3), 2)

        logger.info(f"GPU detected: {gpu_info['device_name']}")
        logger.info(f"GPU memory: {gpu_info['memory_total_gb']} GB")
    else:
        logger.warning("No GPU detected! Inference will be slow on CPU.")

    return gpu_info


def setup_colab_environment(base_path="/content/models", assets_dir="/content/assets"):
    logger.info("=" * 60)
    logger.info("Setting up RVC Colab environment...")
    logger.info("=" * 60)

    status = {"success": True, "errors": []}

    logger.info("\n[1/4] Checking GPU...")
    gpu_info = check_gpu()
    status["gpu"] = gpu_info

    logger.info("\n[2/4] Creating directories...")
    try:
        dirs = {
            "models": Path(base_path) / "models",
            "indices": Path(base_path) / "indices",
            "assets": Path(assets_dir)
        }

        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {path}")

        status["directories"] = {k: str(v) for k, v in dirs.items()}

    except Exception as e:
        error_msg = f"Failed to create directories: {e}"
        logger.error(error_msg)
        status["errors"].append(error_msg)
        status["success"] = False
        return status

    logger.info("\n[3/4] Downloading base models...")
    try:
        download_results = download_base_models(assets_dir)
        status["models_downloaded"] = download_results

        if not all(download_results.values()):
            error_msg = "Some models failed to download"
            logger.warning(error_msg)
            status["errors"].append(error_msg)

    except Exception as e:
        error_msg = f"Failed to download models: {e}"
        logger.error(error_msg)
        status["errors"].append(error_msg)
        status["success"] = False
        return status

    logger.info("\n[4/4] Setting environment variables...")
    try:
        env_vars = {
            "weight_root": str(dirs["models"]),
            "index_root": str(dirs["indices"]),
            "hubert_path": str(Path(assets_dir) / "hubert_base.pt"),
            "rmvpe_root": str(assets_dir)
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"  {key} = {value}")

        status["environment"] = env_vars

    except Exception as e:
        error_msg = f"Failed to set environment: {e}"
        logger.error(error_msg)
        status["errors"].append(error_msg)
        status["success"] = False
        return status

    logger.info("\n" + "=" * 60)
    if status["success"]:
        logger.info("Setup completed successfully!")
        logger.info(f"GPU: {gpu_info['device_name'] or 'Not available'}")
        logger.info(f"Models directory: {dirs['models']}")
    else:
        logger.error("Setup completed with errors!")
        for error in status["errors"]:
            logger.error(f"  - {error}")
    logger.info("=" * 60)

    return status


def verify_installation():
    results = {
        "torch": False,
        "cuda": False,
        "hubert": False,
        "rmvpe": False,
        "directories": False
    }

    try:
        import torch
        results["torch"] = True
        results["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    hubert_path = os.getenv("hubert_path")
    if hubert_path and Path(hubert_path).exists():
        results["hubert"] = True

    rmvpe_root = os.getenv("rmvpe_root")
    if rmvpe_root:
        rmvpe_path = Path(rmvpe_root) / "rmvpe.pt"
        results["rmvpe"] = rmvpe_path.exists()

    weight_root = os.getenv("weight_root")
    index_root = os.getenv("index_root")
    if weight_root and index_root:
        results["directories"] = Path(weight_root).exists() and Path(index_root).exists()

    return results
