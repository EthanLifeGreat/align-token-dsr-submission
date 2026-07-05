# HiFiGAN Vocoder

基于 HiFiGAN 的神经声码器，用于将梅尔频谱转换为波形。

## 目录结构

```
hifigan/
├── ckpt/                  # 模型权重目录
│   └── generator.pt       # 生成器权重（需要自行提取）
├── model.py               # 模型定义
├── inference.py           # 推理 API
├── extract_weights.py     # 权重提取脚本
├── __init__.py            # 包初始化
└── README.md              # 本文档
```

## 安装依赖

```bash
pip install numpy torch
```

## 权重准备

在使用 API 之前，需要从训练好的检查点中提取生成器权重：

```bash
# 从训练检查点提取生成器权重
python extract_weights.py /path/to/model-best.pt --output ckpt/generator.pt

# 或者使用默认输出路径
python extract_weights.py /path/to/model-best.pt
```

提取脚本会自动：
- 提取生成器部分的权重
- 移除 `module.` 前缀（如果存在）
- 仅保存推理所需的权重，大幅减小文件大小

## API 使用

### 基本使用

```python
from utils.Vocoder.hifigan import HiFiGANAPI

# 初始化（使用默认权重路径）
# 初始化时会自动检测模型需要的 mel_bins 数量
vocoder = HiFiGANAPI()
# 输出: [HiFiGAN] Model expects 128 mel bins as input.

# 查看模型需要的 mel_bins 数量
print(f"模型需要 {vocoder.mel_bins} 个 mel bins")

# 准备梅尔频谱（numpy 数组）
# 形状可以是 (time, mel_bins) 或 (mel_bins, time)
# API 会自动判断并转置
mel = np.random.randn(100, 128)  # 100 frames, 128 mel bins

# 推理
wav, sr = vocoder.inference(mel)
print(f"波形形状: {wav.shape}, 采样率: {sr}")
```

### 指定权重路径

```python
from utils.Vocoder.hifigan import HiFiGANAPI

# 指定自定义权重路径
vocoder = HiFiGANAPI(checkpoint_path="/path/to/generator.pt")
```

### 指定设备

```python
# 使用 CPU
vocoder = HiFiGANAPI(device="cpu")

# 使用特定 GPU
vocoder = HiFiGANAPI(device="cuda:0")

# 自动选择（默认行为：优先使用 GPU）
vocoder = HiFiGANAPI()
```

### 批量推理

```python
import numpy as np

# 准备批量的梅尔频谱
# 形状: (batch, mel_bins, time) 或 (batch, time, mel_bins)
mels = np.random.randn(4, 100, 128)  # 4 个样本, 100 frames, 128 mel bins

# 批量推理
wavs, sr = vocoder.inference_batch(mels)
print(f"输出形状: {wavs.shape}")  # (4, time)
```

### 获取模型信息

```python
# 获取采样率
sr = vocoder.sample_rate  # 默认 16000

# 获取 mel_bins 数量
mel_bins = vocoder.mel_bins  # 从权重自动检测
```

### 返回 Torch 张量

```python
# 返回 torch.Tensor 而非 numpy 数组
wav, sr = vocoder.inference(mel, output_numpy=False)
```

### 使用便捷函数

```python
from utils.Vocoder.hifigan import load_vocoder

vocoder = load_vocoder()  # 使用默认设置
```

## API 参考

### `HiFiGANAPI`

#### `__init__(checkpoint_path=None, device=None, remove_weight_norm=True)`

初始化 HiFiGAN 声码器。初始化时会自动从权重中检测模型需要的 mel_bins 数量并打印提示。

**参数：**
- `checkpoint_path` (str, optional): 模型权重路径。默认为 `ckpt/generator.pt`。
- `device` (str, optional): 运行设备。默认自动选择（优先 GPU）。
- `remove_weight_norm` (bool): 是否移除权重归一化以加速推理。默认 `True`。

#### `mel_bins` (property)

返回模型需要的 mel_bins 数量（从权重自动检测）。

#### `inference(mel, output_numpy=True) -> Tuple[ndarray, int]`

将梅尔频谱转换为波形。会自动判断输入形状并验证 mel_bins 数量是否匹配。

**参数：**
- `mel` (np.ndarray): 输入梅尔频谱，形状为 `(time, mel_bins)` 或 `(mel_bins, time)`。
- `output_numpy` (bool): 是否返回 numpy 数组。默认 `True`。

**返回：**
- `waveform` (np.ndarray): 生成的波形，形状为 `(time,)`。
- `sample_rate` (int): 采样率（默认 16000）。

**错误：**
- 如果输入的 mel_bins 数量与模型不匹配，会抛出 `ValueError`。

#### `inference_batch(mels, output_numpy=True) -> Tuple[ndarray, int]`

批量将梅尔频谱转换为波形。会自动判断输入形状并验证 mel_bins 数量是否匹配。

**参数：**
- `mels` (np.ndarray): 输入梅尔频谱，形状为 `(batch, mel_bins, time)` 或 `(batch, time, mel_bins)`。
- `output_numpy` (bool): 是否返回 numpy 数组。默认 `True`。

**返回：**
- `waveforms` (np.ndarray): 生成的波形，形状为 `(batch, time)`。
- `sample_rate` (int): 采样率。

**错误：**
- 如果输入的 mel_bins 数量与模型不匹配，会抛出 `ValueError`。

#### `sample_rate` (property)

返回模型配置的采样率。

### `load_vocoder(checkpoint_path=None, device=None) -> HiFiGANAPI`

便捷函数，加载并返回 HiFiGAN 声码器实例。

## 模型参数

默认配置（mel_bins 从权重自动检测）：
- 输入通道数（mel bins）：从权重自动检测
- 采样率：16000
- 上采样因子：(10, 4, 4, 2) → 总上采样因子 = 320
- 上采样初始通道数：512

## 注意事项

1. **自动检测 mel_bins**：初始化时会自动从权重中检测模型需要的 mel_bins 数量并打印提示
2. **输入形状自动判断**：推理时会自动判断输入形状 `(time, mel_bins)` 或 `(mel_bins, time)` 并正确转置
3. **输入验证**：如果输入的 mel_bins 数量与模型不匹配，会抛出 `ValueError`
4. 输入梅尔频谱应为标准化的浮点数数组
5. 建议在首次加载时调用 `remove_weight_norm=True` 以优化推理速度
