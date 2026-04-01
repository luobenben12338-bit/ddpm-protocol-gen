from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.tabular import BitSquareDataset
from .model import load_runtime


@dataclass(slots=True)
class SamplingResult:
    hex_rows: list[str]
    count: int
    model_path: Path


def sample_hex_rows(
    dataset: BitSquareDataset,
    *,
    model_in: str | Path,
    count: int,
) -> SamplingResult:
    runtime = load_runtime(model_in)
    runtime.unet.eval()
    sampled = runtime.diffusion.sample(batch_size=count, return_all_timesteps=True)
    final_samples = np.array(sampled[-1]) if isinstance(sampled, list) else sampled.detach().cpu().numpy()
    hex_rows = dataset.restore_from_samples(final_samples)
    return SamplingResult(hex_rows=hex_rows, count=len(hex_rows), model_path=Path(model_in))
