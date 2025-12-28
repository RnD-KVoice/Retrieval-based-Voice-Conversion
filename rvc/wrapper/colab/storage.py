import os
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


def init_storage(base_path="/content/models"):
    base = Path(base_path)
    models_dir = base / "models"
    indices_dir = base / "indices"

    models_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)

    os.environ["weight_root"] = str(models_dir)
    os.environ["index_root"] = str(indices_dir)

    return models_dir, indices_dir


def save_model(file_data, filename, base_path="/content/models"):
    models_dir, _ = init_storage(base_path)

    if not filename.endswith('.pth'):
        filename = filename + '.pth'

    file_path = models_dir / filename

    with open(file_path, 'wb') as f:
        f.write(file_data)

    file_size = file_path.stat().st_size

    metadata = {
        "filename": filename,
        "path": str(file_path),
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2)
    }

    try:
        cpt = torch.load(file_path, weights_only=False, map_location="cpu")
        metadata.update({
            "version": cpt.get("version", "v1"),
            "f0": cpt.get("f0", 1),
            "sample_rate": cpt["config"][-1] if "config" in cpt else None,
            "n_speakers": cpt["config"][-3] if "config" in cpt else None
        })
        del cpt
    except Exception as e:
        logger.warning(f"Could not extract metadata: {e}")
        metadata["error"] = str(e)

    logger.info(f"Saved model: {filename} ({metadata['size_mb']} MB)")
    return metadata


def save_index(file_data, filename, base_path="/content/models"):
    _, indices_dir = init_storage(base_path)

    if not filename.endswith('.index'):
        filename = filename + '.index'

    file_path = indices_dir / filename

    with open(file_path, 'wb') as f:
        f.write(file_data)

    file_size = file_path.stat().st_size

    logger.info(f"Saved index: {filename} ({round(file_size / (1024 * 1024), 2)} MB)")

    return {
        "filename": filename,
        "path": str(file_path),
        "size_mb": round(file_size / (1024 * 1024), 2)
    }


def list_models(base_path="/content/models"):
    models_dir, indices_dir = init_storage(base_path)
    models = []

    for pth_file in models_dir.glob("*.pth"):
        file_size = pth_file.stat().st_size
        model_name = pth_file.stem

        index_file = find_index(model_name, base_path)

        models.append({
            "name": model_name,
            "filename": pth_file.name,
            "path": str(pth_file),
            "size_mb": round(file_size / (1024 * 1024), 2),
            "has_index": index_file is not None,
            "index_path": index_file
        })

    models.sort(key=lambda x: x["name"])
    return models


def get_model_path(model_name, base_path="/content/models"):
    models_dir, _ = init_storage(base_path)

    if not model_name.endswith('.pth'):
        model_name = model_name + '.pth'

    model_path = models_dir / model_name

    if not model_path.exists():
        available = [m["filename"] for m in list_models(base_path)]
        raise FileNotFoundError(f"Model '{model_name}' not found. Available: {available}")

    return str(model_path)


def find_index(model_name, base_path="/content/models"):
    _, indices_dir = init_storage(base_path)

    model_name = model_name.replace('.pth', '')

    for index_file in indices_dir.glob("*.index"):
        if model_name in index_file.stem and "trained" not in index_file.stem:
            return str(index_file)

    return None


def delete_model(model_name, base_path="/content/models"):
    models_dir, _ = init_storage(base_path)

    if not model_name.endswith('.pth'):
        model_name = model_name + '.pth'

    model_path = models_dir / model_name

    if not model_path.exists():
        return False

    model_path.unlink()
    logger.info(f"Deleted model: {model_name}")

    model_stem = model_path.stem
    index_path = find_index(model_stem, base_path)
    if index_path:
        Path(index_path).unlink()
        logger.info(f"Deleted index: {index_path}")

    return True


def get_storage_info(base_path="/content/models"):
    models_dir, indices_dir = init_storage(base_path)
    models = list_models(base_path)
    total_size = sum(m["size_mb"] for m in models)
    index_count = len(list(indices_dir.glob("*.index")))

    return {
        "models_count": len(models),
        "indices_count": index_count,
        "total_size_mb": round(total_size, 2),
        "models_dir": str(models_dir),
        "indices_dir": str(indices_dir)
    }
