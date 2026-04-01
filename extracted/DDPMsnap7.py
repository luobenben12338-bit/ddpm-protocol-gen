# ===== cell 1 =====
import dpkt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder

# ===== cell 2 =====
from inspect import isfunction
from functools import partial

%matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import Dataset,DatasetDict
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.animation as animation

# ===== cell 3 =====
from PIL import Image

# ===== cell 4 =====
reverse_transform = Compose([
     #Lambda(lambda t: (t + 1) / 2),
    # Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

# ===== cell 5 =====
def read_pcap_file(file_path):
    packets = []
    with open(file_path, 'rb') as pcap_file:
        pcap = dpkt.pcap.Reader(pcap_file)
        for timestamp, buf in pcap:
            byte_string = ''.join(format(byte, '02x') for byte in buf)  # 将每个字节转化为十六进制字符串
            packets.append(list(byte_string))  # 将每个字符添加到列表中

    # 创建pandas数据帧
    df = pd.DataFrame(packets)
    return df

# 使用示例
pcap_file_path = 'D:\A-English\pcap\c1\snap7readq.pcap'
packet_dfzero = read_pcap_file(pcap_file_path)
packet_dfzero = packet_dfzero.fillna('0')
packet_df = packet_dfzero

#截断，保留协议应用层s7标识符ox32
tcp = 88
packet_df = packet_df.iloc[:, tcp:]

packet_df

# ===== cell 6 =====
packet_dfzero = packet_dfzero.fillna('0')
packet_df = packet_dfzero

#截断，保留协议应用层s7标识符ox32
tcp = 88
packet_df = packet_df.iloc[:, tcp:]

packet_df

# ===== cell 7 =====
df = np.array(packet_df)
df = pd.DataFrame(df)

# ===== cell 8 =====
#chatgpt改进的代码，让它给出正确答案不容易，90%在乱写代码
def twoencodertensix(data):
    data = np.array(data)
    mapping = {
        '0': '0000', '1': '0001', '2': '0010', '3': '0011',
        '4': '0100', '5': '0101', '6': '0110', '7': '0111',
        '8': '1000', '9': '1001', 'a': '1010', 'b': '1011',
        'c': '1100', 'd': '1101', 'e': '1110', 'f': '1111'
    }
  
    encoded_data = []
    for row in data:
        row_encoded = ''.join([mapping[c] for c in row])
        encoded_data.append(list(row_encoded))
    
    
    df= pd.DataFrame(np.array(encoded_data))
    
    return df

# ===== cell 9 =====
one = twoencodertensix(df)

# ===== cell 10 =====
def arrencoder(data):
    
    #a = np.sqrt(data.shape[1])
    #a = math.ceil(a)
    
    #b = a*a-one.shape[1]

    sqrt_num = math.sqrt(data.shape[1])

    if sqrt_num.is_integer():
        result = int(sqrt_num)
    else:
        result = math.ceil(sqrt_num)
        #if result % 2 != 0:
            #result += 1
        if result % 4 != 0:
            result = result + (4 - result % 4)
    
    #b = result*result-one.shape[1]
    b = result*result-one.shape[1]
    # 获取当前的列名列表
    columns = list(data.columns)

    # 计算新列的数量
    num_new_columns = b

    # 自定义新列名
    new_columns = [f"{i}" for i in range(data.shape[1], num_new_columns+data.shape[1])]

    # 添加多个新列，并指定自定义的列名
    df = one.assign(**{new_columns[i]: '0' for i in range(num_new_columns)})

    return df

# ===== cell 11 =====
ones = arrencoder(one)

# ===== cell 12 =====
ones

# ===== cell 13 =====
onesdf = np.array(ones)

# ===== cell 14 =====
onesdf = onesdf.astype(int)

# ===== cell 15 =====
onesdf

# ===== cell 16 =====
data = onesdf
data = data*2-1
data = data.reshape(data.shape[0],1,int(np.sqrt(data.shape[1])),int(np.sqrt(data.shape[1])))
data.shape

# ===== cell 17 =====
image_size = (data.shape[2])
channels = 1
batch_size = 128
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
batch = next(iter(dataloader))

# ===== cell 18 =====
def exists(x):
    return x is not None
 
# 有val时返回val，val为None时返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
 
# 残差模块，将输入加到输出上
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
 
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
 
# 上采样（反卷积）
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
 
# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

# ===== cell 19 =====
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ===== cell 20 =====
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
 
    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
 
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
 
        x = self.act(x)
        return x
 
class ResnetBlock(nn.Module):
    """Deep Residual Learning for Image Recognition"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
 
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
 
    def forward(self, x, time_emb=None):
        h = self.block1(x)
 
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h
 
        h = self.block2(h)
        return h + self.res_conv(x)
    
class ConvNextBlock(nn.Module):
    """A ConvNet for the 2020s"""
 
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
 
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
 
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
 
    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
 
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
 
        h = self.net(h)
        return h + self.res_conv(x)

# ===== cell 21 =====
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
 
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
 
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
 
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
 
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
 
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))
 
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
 
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
 
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
 
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

# ===== cell 22 =====
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# ===== cell 23 =====
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()
 
        # determine dimensions
        self.channels = channels
 
        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
 
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)
 
        # time embeddings
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
 
        # layers
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
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )
 
    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []
 
        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
 
        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
 
        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
 
        return self.final_conv(x)

# ===== cell 24 =====
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
 
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
 
def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
 
def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# ===== cell 25 =====
timesteps = 200
 
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)
 
# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
 
# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
 
# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
 
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# ===== cell 26 =====
# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
 
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
 
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
 
def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)
 
    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())
 
    return noisy_image

# ===== cell 27 =====
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    # 先采样噪声
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # 用采样得到的噪声去加噪图片
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    
    # 根据加噪了的图片去预测采样的噪声
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
 
    return loss

# ===== cell 28 =====
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
 
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
 
# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
 
    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
 
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs
 
@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

# ===== cell 29 =====
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
 
results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

# ===== cell 30 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
 
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)
 
optimizer = Adam(model.parameters(), lr=1e-3)

# ===== cell 31 =====
epochs = 2
save_interval = 5
 
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()
 
      batch_size = batch.shape[0]
      batch = batch.to(torch.float32);batch = batch.to(device)
 
      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()
 
      loss = p_losses(model, batch, t, loss_type="huber")
 
      if step % 100 == 0:
        print("Loss:", loss.item())
 
      loss.backward()
      optimizer.step()
 
      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
        
      # save model parameters every 5 epochs
      if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

# ===== cell 32 =====
torch.save(model.state_dict(), f'./DDPMsnap7.pth')

# ===== cell 33 =====
model.load_state_dict(torch.load('./DDPMsnap7.pth'))
model.eval()

# ===== cell 34 =====
samples_size = 10

# ===== cell 35 =====
samples = sample(model, image_size=image_size, batch_size=samples_size, channels=channels)

# ===== cell 36 =====
def intb(data,row):
    dtf = pd.DataFrame()
    dtf = pd.DataFrame([(data[i].map(str) + data[i+1].map(str) + data[i+2].map(str) + data[i+3].map(str))if i % 4 == 0 else None for i in range(0, row, 4)]).T
    
    return dtf

# ===== cell 37 =====
#已改进代码
def df_to_tensix(data):
    
    replace_dict = {
        '0000': '0',
        '0001': '1',
        '0010': '2',
        '0011': '3',
        '0100': '4',
        '0101': '5',
        '0110': '6',
        '0111': '7',
        '1000': '8',
        '1001': '9',
        '1010': 'a',
        '1011': 'b',
        '1100': 'c',
        '1101': 'd',
        '1110': 'e',
        '1111': 'f'
    }
    
    for key, value in replace_dict.items():
        data.replace(key, value, inplace=True)
    
    return data

# ===== cell 38 =====
def outsamples(data):
    
    intput0 = data
    intput1 = intput0.reshape(samples_size,data.shape[2]*data.shape[2])
    intput2 = (np.round(intput1)+1)/2
    intput2 = intput2.astype("i8")
    intputdf = pd.DataFrame(intput2)
    intputdf = intputdf.astype(object)
    dftwo = intb(intputdf,data.shape[2]*data.shape[2])
    dtsix = df_to_tensix(dftwo)
    dtsixs = dtsix.iloc[:,:packet_df.shape[1]]
    
    return dtsixs

# ===== cell 39 =====
output = outsamples(samples[timesteps-1])
output

# ===== cell 40 =====
zero = packet_dfzero.iloc[:samples_size, : tcp]
zero

# ===== cell 41 =====
pcap = pd.concat([zero,output],axis=1, ignore_index=True)

# ===== cell 42 =====
pcap

# ===== cell 43 =====
def write_pcap_file(df, file_path):
    with open(file_path, 'wb') as pcap_file:
        pcap_writer = dpkt.pcap.Writer(pcap_file)

        for _, row in df.iterrows():
            byte_string = bytes.fromhex(''.join(row))  # 将十六进制字符串转换为字节串
            pcap_writer.writepkt(byte_string)

# ===== cell 44 =====
write_pcap_file(pcap, 'D:\Runproject\pcapsnap7.pcap')

# ===== cell 45 =====
pd.set_option('display.max_columns',20) # 最大显示列数

# ===== cell 46 =====
output

# ===== cell 47 =====
import binascii
import pandas as pd

# 假设output是一个DataFrame对象，包含多个列

# 合并每一行的元素
merged_rows = output.apply(lambda row: ''.join(str(x) for x in row), axis=1)

# 对每个元素执行unhexlify()函数
data = merged_rows.apply(binascii.unhexlify)

# 打印结果
print(data)

# ===== cell 48 =====
import socket

# 定义需要发送的五个报文
messages = [
    b'\x03\x00\x00\x16\x11\xe0\x00\x00\x00\x01\x00\xc0\x01\n\xc1\x02\x01\x00\xc2\x02\x01\x02',
    b'\x03\x00\x00\x19\x02\xf0\x802\x01\x00\x00\x00\x00\x00\x08\x00\x00\xf0\x00\x00\x01\x00\x01\x01\xe0',
    b'\x03\x00\x00!\x02\xf0\x802\x07\x00\x00\x01\x00\x00\x08\x00\x08\x00\x01\x12\x04\x11D\x01\x00\xff\t\x00\x04\x00\x11\x00\x00',
    b'\x03\x00\x00!\x02\xf0\x802\x07\x00\x00\x02\x00\x00\x08\x00\x08\x00\x01\x12\x04\x11D\x01\x00\xff\t\x00\x04\x00\x1c\x00\x00',
    b'\x03\x00\x00!\x02\xf0\x802\x07\x00\x00\x03\x00\x00\x08\x00\x08\x00\x01\x12\x04\x11D\x01\x00\xff\t\x00\x04\x011\x00\x01'
    
]

messages1 =data_list

# 创建TCP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接到对方主机和端口
host = '192.168.109.1'
port = 102
sock.connect((host, port))

# 依次发送五个报文并接收对方的响应
for message in messages:
    # 发送报文
    sock.sendall(message)

    # 接收响应
    response = sock.recv(1024)

    # 打印响应
    print('接收到响应:', response)
    
    
for message in messages1:
    # 发送报文
    sock.sendall(message)

    # 接收响应
    response = sock.recv(1024)

    # 打印响应
    print('接收到响应:', response)
    
    
# 关闭套接字
sock.close()

# ===== cell 49 =====
data_list = data.tolist()
print(data_list)

# ===== cell 50 =====

