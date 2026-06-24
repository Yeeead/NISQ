from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch


def save_checkpoint(state: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Union[str, Path], map_location=None) -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
