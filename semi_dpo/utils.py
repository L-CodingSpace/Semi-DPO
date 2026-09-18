"""Small helpers shared by the command line tools."""

from __future__ import annotations

import gc
import logging
import os
from contextlib import contextmanager

import torch


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )


def clear_cache(logger=None, message: str = "Cleared cache") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if logger is not None:
            allocated = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"{message} | mem {allocated:.2f}GB | peak {peak:.2f}GB")


def weight_dtype_for(mixed_precision: str | None) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision or "no", torch.float32)


@contextmanager
def adapters_disabled(model: torch.nn.Module):
    """Temporarily bypass LoRA adapters, also on FSDP-wrapped models."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    layers = [m for m in model.modules() if isinstance(m, (BaseTunerLayer, ModulesToSaveWrapper))]
    for layer in layers:
        layer._disable_adapters = True
    try:
        yield
    finally:
        for layer in layers:
            layer._disable_adapters = False


def checkpoint_step(path: str) -> int:
    """``.../checkpoint-1200`` -> ``1200``."""
    name = os.path.basename(os.path.normpath(path))
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Expected a directory named checkpoint-<step>, got {path}")
    return int(name.split("-")[1])
