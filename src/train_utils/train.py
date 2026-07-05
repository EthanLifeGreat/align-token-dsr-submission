"""
Unified Training Script

Usage:
    python src/train.py --model <model_name> --model-config <path> --train-config <path> --output-dir <dir> --per-device-batch-size <n>
"""
import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.trainer_pt_utils import AcceleratorConfig

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


# ============================================
# 模型注册表
# ============================================
MODEL_REGISTRY = {
    "phoneme2mel": {
        "factory": "src.phoneme2mel.factory",
    },
    "phoneme2token": {
        "factory": "src.phoneme2token.factory",
    },
    "wav2phoneme": {
        "factory": "src.wav2phoneme.factory",
    },
}


def is_rank_zero() -> bool:
    """检查是否是 rank 0（用于 DDP 模式）"""
    local_rank = os.environ.get("LOCAL_RANK", "0")
    return local_rank == "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name (e.g., phoneme2mel, wav2phoneme)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config.yaml"
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to train_config.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        required=True,
        help="Per-device sample batch size or token budget, depending on train_config.batch_size_unit"
    )
    return parser.parse_args()


def load_factory(model_name: str):
    """动态加载模型工厂模块"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    factory_module = importlib.import_module(MODEL_REGISTRY[model_name]["factory"])
    return factory_module


def main():
    args = parse_args()

    # 动态加载工厂模块
    factory = load_factory(args.model)

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent

    # 加载配置
    model_config = factory.load_model_config(Path(args.model_config))
    train_config = factory.load_train_config(Path(args.train_config))

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    seed = getattr(train_config, 'seed', 42)
    torch.manual_seed(seed)

    use_bf16 = bool(getattr(train_config, 'bf16', False))
    use_fp16 = bool(getattr(train_config, 'fp16', False)) and not use_bf16
    use_tf32 = bool(getattr(train_config, 'tf32', False))
    batch_size_unit = str(getattr(train_config, 'batch_size_unit', 'sample')).lower()
    if batch_size_unit not in {'sample', 'token'}:
        raise ValueError(
            f"Unsupported batch_size_unit={batch_size_unit}, expected 'sample' or 'token'"
        )
    eval_batch_size_unit = getattr(train_config, "eval_batch_size_unit", None)
    if eval_batch_size_unit is None and batch_size_unit == "token":
        # token batching 的 eval 在 DDP 下更容易触发 remainder 不同步，默认回退到 sample batching。
        eval_batch_size_unit = "sample"
    eval_batch_size_unit = str(eval_batch_size_unit or batch_size_unit).lower()
    if eval_batch_size_unit not in {"sample", "token"}:
        raise ValueError(
            f"Unsupported eval_batch_size_unit={eval_batch_size_unit}, expected 'sample' or 'token'"
        )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        if use_tf32:
            torch.set_float32_matmul_precision("high")

    logger.info(f"Model: {args.model}")
    logger.info(f"Experiment: {train_config.exp_name}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Dataset: {train_config.dataset}")
    logger.info(
        "Precision config: "
        f"fp16={use_fp16}, bf16={use_bf16}, tf32={use_tf32}"
    )
    if batch_size_unit == "sample":
        logger.info(f"Batch size unit: sample, per_device_batch_size={args.per_device_batch_size}")
    else:
        logger.info(
            f"Batch size unit: token, per_device_token_budget={args.per_device_batch_size}"
        )

    # 保存配置
    factory.save_model_config(model_config, output_dir)
    factory.save_train_config(train_config, output_dir)

    # ============================================
    # 创建数据集
    # ============================================
    logger.info("Loading datasets...")
    train_dataset, eval_dataset = factory.get_datasets(model_config, train_config, project_root)
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # ============================================
    # 创建 Collator
    # ============================================
    collator = factory.get_collator(model_config, train_config)

    # ============================================
    # 创建模型
    # ============================================
    logger.info("Building model...")
    model = factory.get_model(model_config, train_config, train_dataset)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # ============================================
    # 加载预训练权重（微调模式）
    # ============================================
    finetune_checkpoint = getattr(train_config, 'finetune_from_checkpoint', None)
    if finetune_checkpoint:
        logger.info(f"Loading pretrained weights from: {finetune_checkpoint}")
        checkpoint_path = Path(finetune_checkpoint)

        # 支持两种格式：safetensors 和 pytorch
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            logger.info(f"Loaded weights from safetensors: {safetensors_path}")
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu")
            logger.info(f"Loaded weights from pytorch_model.bin: {pytorch_path}")
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

        # 加载权重 (strict=False 允许部分匹配)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")
        logger.info("Pretrained weights loaded successfully!")

    # ============================================
    # 训练参数
    # ============================================
    dataloader_num_workers = getattr(train_config, 'num_workers', 4)
    dataloader_pin_memory = getattr(train_config, 'pin_memory', True)
    dataloader_persistent_workers = (
        getattr(train_config, 'persistent_workers', False) and dataloader_num_workers > 0
    )
    dataloader_prefetch_factor = getattr(train_config, 'prefetch_factor', None)
    if dataloader_num_workers == 0:
        dataloader_prefetch_factor = None

    logger.info(
        "Dataloader config: "
        f"num_workers={dataloader_num_workers}, "
        f"pin_memory={dataloader_pin_memory}, "
        f"persistent_workers={dataloader_persistent_workers}, "
        f"prefetch_factor={dataloader_prefetch_factor}"
    )

    train_batch_size_for_hf = args.per_device_batch_size if batch_size_unit == "sample" else 1
    if eval_batch_size_unit == "sample":
        # 当训练为 token budget 时，这里需要一个 per-device 的 sample batch size。
        eval_batch_size_for_hf = int(getattr(train_config, "per_device_eval_batch_size", None) or 1)
    else:
        eval_batch_size_for_hf = train_batch_size_for_hf

    accelerator_config = AcceleratorConfig(
        even_batches=False if batch_size_unit == "token" else True
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.num_epochs,
        max_steps=getattr(train_config, 'max_steps', -1),
        per_device_train_batch_size=train_batch_size_for_hf,
        per_device_eval_batch_size=eval_batch_size_for_hf,
        gradient_accumulation_steps=getattr(train_config, 'accumulation_steps', 1),
        learning_rate=train_config.learning_rate,
        warmup_ratio=getattr(train_config, 'warmup_ratio', 0.0),
        lr_scheduler_type=getattr(train_config, 'lr_scheduler_type', 'linear'),
        logging_dir=str(output_dir / "tensorboard"),
        logging_steps=getattr(train_config, 'logging_steps', 20),
        save_strategy="steps",
        save_steps=train_config.save_steps,
        save_total_limit=getattr(train_config, 'save_total_limit', 2),
        eval_strategy="steps",
        eval_steps=train_config.save_steps,
        eval_accumulation_steps=getattr(train_config, 'eval_accumulation_steps', 100),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=use_tf32,
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        remove_unused_columns=False,
        report_to=[],
        max_grad_norm=getattr(train_config, 'grad_clip', 1.0),
        prediction_loss_only=True,  # 评估时只计算 loss，不保存预测结果
        # Turning this on adds overhead each step. Only enable when you truly have
        # parameters that are intermittently unused in the forward pass.
        ddp_find_unused_parameters=getattr(train_config, "ddp_find_unused_parameters", False),
        accelerator_config=accelerator_config,
    )
    training_args.per_device_train_token_budget = (
        args.per_device_batch_size if batch_size_unit == "token" else None
    )
    training_args.per_device_eval_token_budget = (
        args.per_device_batch_size if (batch_size_unit == "token" and eval_batch_size_unit == "token") else None
    )
    training_args.eval_batch_size_unit = eval_batch_size_unit

    # ============================================
    # 创建 Trainer
    # ============================================
    trainer = factory.get_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        model_config=model_config,
        train_config=train_config,
    )

    # ============================================
    # 断点续训
    # ============================================
    checkpoint = None
    resume = getattr(train_config, 'resume', False)
    if resume:
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[1])))
            logger.info(f"Resuming from checkpoint: {checkpoint}")

    # ============================================
    # 开始训练
    # ============================================
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # ============================================
    # 保存最终模型
    # ============================================
    logger.info("Saving final model...")
    trainer.save_model(str(output_dir / "final"))

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
