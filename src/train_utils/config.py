"""
通用训练配置类

统一 wav2phoneme 和 phoneme2mel 的训练配置
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import yaml
from pathlib import Path
from transformers import PretrainedConfig


@dataclass
class TrainConfig:
    """通用训练配置 (train_config.yaml)"""
    # ============================================
    # 数据配置（相对于仓库根目录）
    # ============================================
    data_dir: str = "data/"
    dataset: Union[str, List[str]] = "AISHELL-2"  # 单个数据集名或多个数据集名列表
    source_dataset: Optional[Union[str, List[str]]] = None  # paired/SFT: 输入音频来源数据集
    source_data_dir: Optional[str] = None  # paired/SFT: 输入音频数据根目录；默认跟随 data_dir
    target_manifest_csv: str = "manifests/synth_manifest.csv"  # paired/SFT: 合成目标 manifest
    train_splits: Optional[Union[str, List[str]]] = None  # 默认 train；可设为 [train, dev]
    eval_splits: Optional[Union[str, List[str]]] = None  # 默认 dev
    valid_frames_csv: Optional[str] = None  # 数据过滤csv
    valid_data_csv: Optional[str] = None  # token2mel: 验证数据csv（暂不支持）

    # ============================================
    # 训练参数
    # ============================================
    learning_rate: float = 1e-5
    effective_batch_size: int = 8  # 总有效 batch，单位由 batch_size_unit 决定
    batch_size_unit: str = "sample"  # sample / token
    # 评估 batch 的单位；默认 None 表示跟随 batch_size_unit。
    # 经验上 token-based eval 在 DDP 下更容易出现 remainder/同步问题，因此当训练为 token 时，
    # 如果用户不显式指定，这里由上层脚本选择一个更安全的默认值。
    eval_batch_size_unit: Optional[str] = None  # sample / token / None
    # 当 eval_batch_size_unit=sample 时使用（per-device）。
    per_device_eval_batch_size: Optional[int] = None
    batch_size: Optional[int] = None  # 兼容旧的 batch_size 配置
    accumulation_steps: int = 1
    num_epochs: int = 10
    max_steps: int = -1
    train_max_samples: Optional[int] = None
    eval_max_samples: Optional[int] = None
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"  # linear / cosine / constant
    grad_clip: Optional[float] = 1.0

    # ============================================
    # 保存配置
    # ============================================
    exp_name: str = "exp_default"
    output_dir: str = "results/${exp_name}"  # 由具体模块覆盖
    save_steps: int = 1000
    save_total_limit: int = 2

    # ============================================
    # 日志配置
    # ============================================
    logging_steps: int = 20
    eval_accumulation_steps: int = 100

    # ============================================
    # 其他配置
    # ============================================
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    tf32: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    length_bucketing: bool = False
    length_bucket_multiplier: int = 50
    resume: bool = False
    # DDP: 默认关闭 find_unused_parameters，避免每步额外 autograd graph 遍历开销。
    # 如果模型存在条件分支导致某些迭代确实会出现未参与反传的参数，可在 YAML 中显式设为 true。
    ddp_find_unused_parameters: bool = False

    # ============================================
    # 特定模块配置
    # ============================================
    # wav2phoneme: Eval 子集大小（解决 OOM 问题）
    eval_subset_size: Optional[int] = None

    # 微调配置
    finetune_from_checkpoint: Optional[str] = None

    # token2mel: 最大长度（超过则随机截断）
    max_len: Optional[int] = None


def save_yaml(config: object, path: Path) -> None:
    """保存配置到 YAML 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(config, PretrainedConfig):
            data = config.to_dict()
        else:
            data = asdict(config)
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(config_class: type, path: Path) -> object:
    """从 YAML 文件加载配置"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def save_train_config(config: TrainConfig, output_dir: Path) -> None:
    """保存训练配置"""
    save_yaml(config, Path(output_dir) / "train_config.yaml")


def load_train_config(path: Path) -> TrainConfig:
    """加载训练配置"""
    return load_yaml(TrainConfig, path)
