import os
import pickle

def save_lmm_checkpoint(params, path="checkpoints/lmm_latest.pkl", metadata=None):
    """Saves LMM PyTree parameters and metadata to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"params": params, "metadata": metadata or {}}
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_lmm_checkpoint(path="checkpoints/lmm_latest.pkl"):
    """Loads LMM PyTree parameters from disk. Returns params dict or None."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    return data

def get_checkpoint_metadata(path="checkpoints/lmm_latest.pkl"):
    """Returns checkpoint metadata dict, or empty dict if not available."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "metadata" in data:
        return data["metadata"]
    return {}
