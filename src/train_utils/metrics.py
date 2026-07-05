"""
通用 Metrics 模块

提供通用的 MetricsCallback 和分布式辅助函数
"""
import os
import logging
from typing import Dict, List, Optional, Callable, Any

import torch
import torch.distributed as dist
import numpy as np
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


# ============================================
# 分布式辅助函数
# ============================================

def is_rank_zero() -> bool:
    """检查是否是 rank 0（用于 DDP 模式）"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def all_reduce_mean(value: float) -> float:
    """在所有 rank 之间计算平均值（DDP 模式）"""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()

def all_reduce_sums_counts(
    sums: List[float],
    counts: List[int],
) -> tuple[list[float], list[int]]:
    """
    在所有 rank 之间做 SUM 规约，返回全局 sums/counts。

    关键点：即使本 rank 没有样本（count=0），也必须参与同一次 collective，
    否则会出现某些 rank 多/少调用一次 dist.all_reduce 导致 NCCL 死锁。
    """
    if not dist.is_initialized():
        return list(sums), list(counts)

    if len(sums) != len(counts):
        raise ValueError(f"sums/counts length mismatch: {len(sums)} vs {len(counts)}")

    device = torch.device("cuda")
    # 用 float32 承载 sum 和 count，保证一次 all_reduce 完成同步。
    t = torch.empty((2, len(sums)), device=device, dtype=torch.float32)
    t[0] = torch.tensor([float(x) for x in sums], device=device, dtype=torch.float32)
    t[1] = torch.tensor([float(int(x)) for x in counts], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    global_sums = t[0].tolist()
    global_counts = [int(round(x)) for x in t[1].tolist()]
    return global_sums, global_counts


# ============================================
# 通用 MetricsCallback
# ============================================

class MetricsCallback(TrainerCallback):
    """
    通用 Metrics Callback

    从模型输出的 outputs["metrics"] 中提取指标并记录。

    工作流程：
    1. 模型 forward 返回包含 "metrics" 字段的输出
    2. Trainer 在 compute_loss 或 prediction_step 中调用 update_metrics(outputs)
    3. Callback 自动记录到日志和 tensorboard

    Example:
        # 模型 forward 返回:
        {
            "loss": loss,
            "mel_pred": ...,
            "metrics": {"loss_after_postnet": 0.5, "loss_before_postnet": 0.6}
        }

        # Trainer 中:
        callback.update_metrics(outputs, is_training=model.training)
    """

    def __init__(
        self,
        metric_names: List[str],
        log_format: Optional[str] = None,
    ):
        """
        Args:
            metric_names: 需要记录的指标名称列表
            log_format: 自定义日志格式字符串
        """
        self.metric_names = metric_names
        self.log_format = log_format

        # 为每个 metric 创建 train/eval 列表
        self.train_metrics: Dict[str, List[float]] = {name: [] for name in metric_names}
        self.eval_metrics: Dict[str, List[float]] = {name: [] for name in metric_names}

        # Tensorboard writer (只在 rank 0 创建)
        self._writer: Optional[SummaryWriter] = None
        self._logging_dir: Optional[str] = None

    def _get_writer(self, logging_dir: str) -> Optional[SummaryWriter]:
        """获取 tensorboard writer（只在 rank 0）"""
        if not is_rank_zero():
            return None

        if self._writer is None or self._logging_dir != logging_dir:
            if self._writer is not None:
                self._writer.close()
            os.makedirs(logging_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=logging_dir)
            self._logging_dir = logging_dir
        return self._writer

    def update_metrics(self, outputs: Dict, is_training: bool) -> None:
        """
        从模型输出中更新指标

        Args:
            outputs: 模型输出，包含 "metrics" 字段
            is_training: 是否是训练阶段
        """
        if "metrics" not in outputs:
            return

        metrics = outputs["metrics"]
        target = self.train_metrics if is_training else self.eval_metrics
        phase = "train" if is_training else "eval"
        
        for name, value in metrics.items():
            if name in target:
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().cpu().item()
                target[name].append(float(value))
                logger.debug(f"[update_metrics] {phase}: added {name}={value}")

    def _clear_metrics(self, is_training: bool) -> None:
        """清理累积的指标"""
        target = self.train_metrics if is_training else self.eval_metrics
        for name in target:
            target[name] = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """日志输出时，处理累积的指标"""
        if logs is None:
            return

        global_step = state.global_step

        # 以 logs 的 key 判断这次 on_log 是训练还是评估日志。这个判断在所有 rank 上一致。
        is_eval_log = any(str(k).startswith("eval_") for k in logs.keys())
        phase = "eval" if is_eval_log else "train"
        metrics_dict = self.eval_metrics if is_eval_log else self.train_metrics

        # 汇总本 rank 的 sum/count，并做全局 all_reduce。
        local_sums: List[float] = []
        local_counts: List[int] = []
        for name in self.metric_names:
            values = metrics_dict[name]
            if values:
                local_sums.append(float(np.sum(values)))
                local_counts.append(int(len(values)))
            else:
                local_sums.append(0.0)
                local_counts.append(0)

        global_sums, global_counts = all_reduce_sums_counts(local_sums, local_counts)
        metric_means: Dict[str, float] = {}
        any_samples = False
        for idx, name in enumerate(self.metric_names):
            cnt = int(global_counts[idx])
            if cnt > 0:
                any_samples = True
                metric_means[name] = float(global_sums[idx]) / float(cnt)

        # 清理本阶段累积，避免跨 log 周期残留
        self._clear_metrics(is_training=not is_eval_log)

        # 如果全局没有任何样本（比如某次 eval 因为 remainder 全部被跳过），不输出也不写 tb
        if any_samples and is_rank_zero() and metric_means:
            writer = self._get_writer(args.logging_dir)

            if self.log_format:
                metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metric_means.items())
                log_msg = self.log_format.format(
                    phase=phase, step=global_step, metrics=metrics_str, **metric_means
                )
            else:
                metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metric_means.items())
                log_msg = f"[{phase}] step={global_step} | {metrics_str}"

            logger.info(log_msg)

            if writer is not None:
                for name, value in metric_means.items():
                    writer.add_scalar(f"{phase}/{name}", value, global_step)

        # 写入 logs 中的其他指标到 tensorboard
        if is_rank_zero():
            writer = self._get_writer(args.logging_dir)
            if writer is not None:
                phase = "eval" if any(k.startswith("eval_") for k in logs) else "train"
                written_keys = set(self.metric_names)
                for key, value in logs.items():
                    if isinstance(value, (int, float)) and key not in written_keys:
                        writer.add_scalar(f"{phase}/{key}", value, global_step)
                writer.flush()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """训练结束时关闭 writer"""
        if self._writer is not None and is_rank_zero():
            self._writer.close()
