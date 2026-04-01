# Protocol Replay Lab
参考
1.The Annotated Diffusion Model
https://huggingface.co/blog/annotated-diffusion

2.DiffusionFuzz: Fuzzing Framework of Industrial Control Protocols Based on Denoising Diffusion Probabilistic Mode
https://ieeexplore.ieee.org/document/10529254

一个面向 [`extracted/ddpmmodbusx02.py`](extracted/ddpmmodbusx02.py)、[`extracted/ddpmmodbusx03.py`](extracted/ddpmmodbusx03.py) 与 [`extracted/DDPMsnap7.py`](extracted/DDPMsnap7.py) 的重构项目。

目标是把原始 Notebook 风格脚本整理为一个适合上传 GitHub 的 Python 工程，并保留三个脚本中的核心能力：

- Modbus 十六进制文本导入、bit 编码、DDPM 训练、采样、导出、发送
- S7 pcap 提取、payload 编码、DDPM 训练、采样、回写、发送
- 统一 CLI 命令入口

## 项目结构

```text
src/protocol_replay_lab/
├─ common/
│  ├─ hexio.py
│  ├─ bitcodec.py
│  ├─ tabular.py
│  ├─ pcap_io.py
│  ├─ sender.py
│  └─ models.py
├─ ddpm/
│  ├─ model.py
│  ├─ trainer.py
│  └─ sampler.py
├─ protocols/
│  ├─ modbus.py
│  └─ s7.py
└─ cli.py
```

## 功能映射

### Modbus

对应 [`extracted/ddpmmodbusx02.py`](extracted/ddpmmodbusx02.py) 与 [`extracted/ddpmmodbusx03.py`](extracted/ddpmmodbusx03.py)：

- 从文本读取 hex 报文
- 转 bit 方阵数据
- 使用真实 UNet/DDPM 训练噪声预测模型
- 采样恢复 hex 报文
- 导出文本结果
- 发送到 Modbus TCP `502` 端口并接收响应

### S7

对应 [`extracted/DDPMsnap7.py`](extracted/DDPMsnap7.py)：

- 从 pcap 中按偏移提取 S7 payload
- 将 payload 转换为 bit 方阵数据
- 使用真实 UNet/DDPM 训练噪声预测模型
- 采样恢复 S7 payload
- 与原始 prefix 拼接后回写 pcap
- 先发送握手报文，再发送业务报文到 `102` 端口

## DDPM 实现说明

当前 [`src/protocol_replay_lab/ddpm/model.py`](src/protocol_replay_lab/ddpm/model.py) 已经迁移原始脚本中的核心网络与扩散过程，包括：

- [`Unet`](src/protocol_replay_lab/ddpm/model.py:225)
- [`Residual`](src/protocol_replay_lab/ddpm/model.py:61)
- [`ResnetBlock`](src/protocol_replay_lab/ddpm/model.py:107)
- [`ConvNextBlock`](src/protocol_replay_lab/ddpm/model.py:123)
- [`Attention`](src/protocol_replay_lab/ddpm/model.py:147)
- [`LinearAttention`](src/protocol_replay_lab/ddpm/model.py:169)
- [`GaussianDiffusion`](src/protocol_replay_lab/ddpm/model.py:353)
- `linear/cosine/quadratic/sigmoid` beta schedule

训练逻辑位于 [`src/protocol_replay_lab/ddpm/trainer.py`](src/protocol_replay_lab/ddpm/trainer.py)，采样逻辑位于 [`src/protocol_replay_lab/ddpm/sampler.py`](src/protocol_replay_lab/ddpm/sampler.py)。

## 安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

开发模式安装：

```bash
pip install -e .
```

## 快速开始

### 1. 训练 Modbus 模型

```bash
python -m protocol_replay_lab.cli modbus train \
  --input samples/modbus_hex.txt \
  --model-out outputs/models/modbus.pt \
  --preview-dir outputs/images/modbus \
  --epochs 2 \
  --batch-size 32 \
  --timesteps 200 \
  --dim 16 \
  --dim-mults 1,2,4 \
  --schedule linear
```

### 2. 采样 Modbus 报文

```bash
python -m protocol_replay_lab.cli modbus sample \
  --input samples/modbus_hex.txt \
  --model-in outputs/models/modbus.pt \
  --count 3 \
  --output outputs/csv/modbus_generated.txt
```

### 3. 发送 Modbus 报文

```bash
python -m protocol_replay_lab.cli modbus send \
  --host 127.0.0.1 \
  --input outputs/csv/modbus_generated.txt
```

### 4. 训练 S7 模型

```bash
python -m protocol_replay_lab.cli s7 train \
  --input samples/s7_hex.txt \
  --model-out outputs/models/s7.pt \
  --preview-dir outputs/images/s7 \
  --epochs 2 \
  --batch-size 32 \
  --timesteps 200 \
  --dim 16 \
  --dim-mults 1,2,4 \
  --schedule linear
```

### 5. 采样 S7 payload

```bash
python -m protocol_replay_lab.cli s7 sample \
  --input samples/s7_hex.txt \
  --model-in outputs/models/s7.pt \
  --count 3 \
  --output outputs/csv/s7_generated.txt
```

### 6. 发送 S7 payload

```bash
python -m protocol_replay_lab.cli s7 send \
  --host 192.168.109.1 \
  --input outputs/csv/s7_generated.txt
```

### 7. 从 pcap 提取 S7 payload

```bash
python -m protocol_replay_lab.cli s7 extract-pcap \
  --input your_s7.pcap \
  --output outputs/csv/s7_payloads.txt \
  --offset 88
```

### 8. 将生成的 S7 payload 回写为 pcap

```bash
python -m protocol_replay_lab.cli s7 write-pcap \
  --source-pcap your_s7.pcap \
  --input outputs/csv/s7_generated.txt \
  --output outputs/pcap/s7_generated.pcap \
  --offset 88
```

## 输出目录

建议输出到以下路径：

- `outputs/models/`：模型文件
- `outputs/images/`：采样预览图
- `outputs/csv/`：生成的十六进制文本
- `outputs/pcap/`：回写后的 pcap

## 注意事项

- S7 pcap 提取默认偏移为 `88` 字节，对应原始脚本中的固定逻辑。
- [`src/protocol_replay_lab/common/sender.py`](src/protocol_replay_lab/common/sender.py) 默认启用超时与失败重连。
- 示例文件 [`samples/modbus_hex.txt`](samples/modbus_hex.txt) 与 [`samples/s7_hex.txt`](samples/s7_hex.txt) 仅用于演示命令结构。
- 运行 CLI 前需要先安装 [`requirements.txt`](requirements.txt) 中的依赖，尤其是 `torch` 与 `dpkt`。
