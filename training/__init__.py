__all__ = [
    "load_checkpoint",
    "run_nisq_training",
    "save_checkpoint",
]


def __getattr__(name: str):
    if name == "run_nisq_training":
        from .train_nisq import run_nisq_training
        return run_nisq_training
    if name == "load_checkpoint":
        from utils.checkpoint import load_checkpoint
        return load_checkpoint
    if name == "save_checkpoint":
        from utils.checkpoint import save_checkpoint
        return save_checkpoint
    raise AttributeError("module 'training' has no attribute '{}'".format(name))
