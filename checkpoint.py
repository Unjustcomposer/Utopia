import os
import pickle

def save_lmm_checkpoint(params, path="checkpoints/lmm_latest.pkl"):
    """Saves LMM PyTree parameters to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)

def load_lmm_checkpoint(path="checkpoints/lmm_latest.pkl"):
    """Loads LMM PyTree parameters from disk."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
