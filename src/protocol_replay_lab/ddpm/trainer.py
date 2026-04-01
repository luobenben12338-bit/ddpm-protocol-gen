from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..common.tabular import BitSquareDataset
from .model import DdpmArtifact, DdpmRuntime, build_runtime, save_tensor_preview


@dataclass(slots=True)
class TrainingResult:
    model_path: Path
    preview_dir: Path
    sample_count: int
    side: int
    epochs: int
    loss_history: list[float] = field(default_factory=list)
    device: str = 'cpu'


def train_echo_ddpm(
    dataset: BitSquareDataset,
    *,
    model_out: str | Path,
    preview_dir: str | Path,
    preview_count: int = 4,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    timesteps: int = 200,
    dim: int = 16,
    dim_mults: tuple[int, ...] = (1, 2, 4),
    schedule: str = 'linear',
    save_every_epoch: bool = True,
) -> TrainingResult:
    tensor_data = dataset.as_tensor_input()
    if tensor_data.size == 0:
        raise ValueError('训练数据为空，无法训练 DDPM')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    artifact = DdpmArtifact(
        side=dataset.side,
        channels=1,
        dim=dim,
        dim_mults=dim_mults,
        timesteps=timesteps,
        schedule=schedule,
    )
    runtime = build_runtime(artifact, device=device)
    runtime.unet.train()
    optimizer = Adam(runtime.unet.parameters(), lr=learning_rate)

    train_tensor = torch.from_numpy(tensor_data.astype(np.float32))
    dataloader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    loss_history: list[float] = []
    model_path = Path(model_out)
    preview_path = Path(preview_dir)

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(torch.float32).to(device)
            t = torch.randint(0, runtime.artifact.timesteps, (batch.shape[0],), device=device).long()
            loss = runtime.diffusion.p_losses(batch, t, loss_type='huber')
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))
        if save_every_epoch:
            runtime.save(model_path)

    runtime.save(model_path)
    runtime.unet.eval()
    preview_steps = runtime.diffusion.sample(batch_size=min(preview_count, max(len(dataset.hex_rows), 1)), return_all_timesteps=True)
    final_preview = np.array(preview_steps[-1]) if isinstance(preview_steps, list) else preview_steps.detach().cpu().numpy()
    save_tensor_preview(final_preview, preview_path, prefix='train-preview')

    return TrainingResult(
        model_path=model_path,
        preview_dir=preview_path,
        sample_count=min(preview_count, max(len(dataset.hex_rows), 1)),
        side=dataset.side,
        epochs=epochs,
        loss_history=loss_history,
        device=device,
    )
