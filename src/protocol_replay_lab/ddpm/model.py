from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor, einsum, nn
import torch.nn.functional as F


LossType = Literal['l1', 'l2', 'huber']
ScheduleType = Literal['linear', 'cosine', 'quadratic', 'sigmoid']


def ensure_torch_available() -> None:
    return None


@dataclass(slots=True)
class DdpmArtifact:
    side: int
    channels: int = 1
    dim: int = 16
    init_dim: int | None = None
    dim_mults: tuple[int, ...] = (1, 2, 4)
    with_time_emb: bool = True
    resnet_block_groups: int = 8
    use_convnext: bool = True
    convnext_mult: int = 2
    timesteps: int = 200
    schedule: ScheduleType = 'linear'
    state_dict: dict | None = None

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload['dim_mults'] = list(self.dim_mults)
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> 'DdpmArtifact':
        return cls(
            side=int(payload['side']),
            channels=int(payload.get('channels', 1)),
            dim=int(payload.get('dim', 16)),
            init_dim=payload.get('init_dim'),
            dim_mults=tuple(payload.get('dim_mults', [1, 2, 4])),
            with_time_emb=bool(payload.get('with_time_emb', True)),
            resnet_block_groups=int(payload.get('resnet_block_groups', 8)),
            use_convnext=bool(payload.get('use_convnext', True)),
            convnext_mult=int(payload.get('convnext_mult', 2)),
            timesteps=int(payload.get('timesteps', 200)),
            schedule=payload.get('schedule', 'linear'),
            state_dict=payload.get('state_dict'),
        )


def exists(value):
    return value is not None


def default(value, default_value):
    if exists(value):
        return value
    return default_value() if callable(default_value) else default_value


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int) -> nn.Module:
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim: int) -> nn.Module:
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor | None = None) -> Tensor:
        h = self.block1(x)
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, 'b c -> b c 1 1') + h
        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, mult: int = 2, norm: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor | None = None) -> Tensor:
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')
        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, _channels, height, width = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=height, y=width)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: Tensor) -> Tensor:
        batch, _channels, height, width = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=height, y=width)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        channels: int = 3,
        with_time_emb: bool = True,
        resnet_block_groups: int = 8,
        use_convnext: bool = True,
        convnext_mult: int = 2,
    ):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = (
            lambda dim_in, dim_out, time_emb_dim=None: ConvNextBlock(dim_in, dim_out, time_emb_dim=time_emb_dim, mult=convnext_mult)
            if use_convnext
            else ResnetBlock(dim_in, dim_out, time_emb_dim=time_emb_dim, groups=resnet_block_groups)
        )
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h: list[Tensor] = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int) -> Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int) -> Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int) -> Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def build_beta_schedule(timesteps: int, schedule: ScheduleType) -> Tensor:
    if schedule == 'linear':
        return linear_beta_schedule(timesteps)
    if schedule == 'cosine':
        return cosine_beta_schedule(timesteps)
    if schedule == 'quadratic':
        return quadratic_beta_schedule(timesteps)
    if schedule == 'sigmoid':
        return sigmoid_beta_schedule(timesteps)
    raise ValueError(f'不支持的 beta schedule: {schedule}')


def extract(values: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    batch_size = t.shape[0]
    out = values.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianDiffusion(nn.Module):
    def __init__(self, model: Unet, *, image_size: int, channels: int = 1, timesteps: int = 200, schedule: ScheduleType = 'linear'):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.schedule = schedule

        betas = build_beta_schedule(timesteps=timesteps, schedule=schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None, loss_type: LossType = 'huber') -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        if loss_type == 'l1':
            return F.l1_loss(noise, predicted_noise)
        if loss_type == 'l2':
            return F.mse_loss(noise, predicted_noise)
        if loss_type == 'huber':
            return F.smooth_l1_loss(noise, predicted_noise)
        raise ValueError(f'不支持的 loss_type: {loss_type}')

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor, t_index: int) -> Tensor:
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Sequence[int], return_all_timesteps: bool = True) -> list[np.ndarray] | Tensor:
        device = next(self.model.parameters()).device
        batch = shape[0]
        img = torch.randn(tuple(shape), device=device)
        imgs: list[np.ndarray] = []
        for step in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((batch,), step, device=device, dtype=torch.long), step)
            if return_all_timesteps:
                imgs.append(img.detach().cpu().numpy())
        return imgs if return_all_timesteps else img

    @torch.no_grad()
    def sample(self, batch_size: int = 16, return_all_timesteps: bool = True) -> list[np.ndarray] | Tensor:
        return self.p_sample_loop(
            shape=(batch_size, self.channels, self.image_size, self.image_size),
            return_all_timesteps=return_all_timesteps,
        )


@dataclass(slots=True)
class DdpmRuntime:
    artifact: DdpmArtifact
    unet: Unet
    diffusion: GaussianDiffusion

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.artifact.to_payload()
        payload['state_dict'] = self.unet.state_dict()
        torch.save(payload, file_path)


def build_runtime(artifact: DdpmArtifact, device: str | torch.device | None = None) -> DdpmRuntime:
    unet = Unet(
        dim=artifact.dim,
        init_dim=artifact.init_dim,
        out_dim=artifact.channels,
        dim_mults=artifact.dim_mults,
        channels=artifact.channels,
        with_time_emb=artifact.with_time_emb,
        resnet_block_groups=artifact.resnet_block_groups,
        use_convnext=artifact.use_convnext,
        convnext_mult=artifact.convnext_mult,
    )
    diffusion = GaussianDiffusion(
        unet,
        image_size=artifact.side,
        channels=artifact.channels,
        timesteps=artifact.timesteps,
        schedule=artifact.schedule,
    )
    if artifact.state_dict is not None:
        unet.load_state_dict(artifact.state_dict)
    if device is not None:
        diffusion = diffusion.to(device)
    return DdpmRuntime(artifact=artifact, unet=unet, diffusion=diffusion)


def load_runtime(path: str | Path, device: str | torch.device | None = None) -> DdpmRuntime:
    payload = torch.load(Path(path), map_location=device or 'cpu')
    artifact = DdpmArtifact.from_payload(payload)
    return build_runtime(artifact, device=device)


def save_tensor_preview(tensor_batch: np.ndarray | Tensor, output_dir: str | Path, prefix: str = 'sample') -> list[Path]:
    file_dir = Path(output_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(tensor_batch, Tensor):
        array_batch = tensor_batch.detach().cpu().numpy()
    else:
        array_batch = tensor_batch
    outputs: list[Path] = []
    for index, sample in enumerate(array_batch):
        array = sample[0] if sample.ndim == 3 else sample
        normalized = ((array + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        target = file_dir / f'{prefix}-{index:03d}.png'
        Image.fromarray(normalized, mode='L').save(target)
        outputs.append(target)
    return outputs
