import os
import pickle

# Base directory is 2 levels up from utopia/core/checkpoint.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)

def save_lmm_checkpoint(params, path="checkpoints/lmm_latest.pkl", metadata=None):
    """Saves LMM PyTree parameters and metadata to disk."""
    path = _resolve_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"params": params, "metadata": metadata or {}}
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_lmm_checkpoint(path="checkpoints/lmm_latest.pkl"):
    """Loads LMM PyTree parameters from disk. Returns params dict or None."""
    path = _resolve_path(path)
    if not os.path.exists(path):
        import logging
        logging.warning(f"WARNING: Checkpoint not found at {path}. Initializing LMM with untrained random weights.")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    return data

def get_checkpoint_metadata(path="checkpoints/lmm_latest.pkl"):
    """Returns checkpoint metadata dict, or empty dict if not available."""
    path = _resolve_path(path)
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "metadata" in data:
        return data["metadata"]
    return {}
