"""
通用 BaseTrainer

支持模型 forward 返回 loss 和 metrics 的通用 Trainer
"""
import logging
from functools import partial
from typing import Dict, Optional, Any, List, Sequence

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer import is_datasets_available, seed_worker

try:
    import datasets
except ImportError:  # pragma: no cover
    datasets = None

from src.train_utils.metrics import MetricsCallback
from src.train_utils.bucket_sampler import (
    DistributedLengthBucketBatchSampler,
    DistributedTokenBucketBatchSampler,
)


logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):
    """
    通用 Trainer

    要求模型 forward 支持 return_loss=True 参数：
    - return_loss=False: 返回 (pred, ...) 或 dict
    - return_loss=True: 返回 {"loss": loss, "metrics": {...}, ...}

    自动处理：
    - compute_loss: 调用 model(..., return_loss=True)
    - metrics 记录: 通过 MetricsCallback 从 outputs["metrics"] 提取
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        metric_names: List[str] = None,
        train_lengths: Optional[Sequence[int]] = None,
        eval_lengths: Optional[Sequence[int]] = None,
        length_bucket_multiplier: int = 50,
        batch_size_unit: str = "sample",
        eval_batch_size_unit: Optional[str] = None,
        **kwargs,
    ):
        # 创建 metrics callback
        self.metric_names = metric_names or []
        self.metrics_callback = MetricsCallback(metric_names=self.metric_names)
        self.train_lengths = list(train_lengths) if train_lengths is not None else None
        self.eval_lengths = list(eval_lengths) if eval_lengths is not None else None
        self.length_bucket_multiplier = int(length_bucket_multiplier)
        self.batch_size_unit = str(batch_size_unit).lower()
        if self.batch_size_unit not in {"sample", "token"}:
            raise ValueError(
                f"Unsupported batch_size_unit={batch_size_unit}, expected 'sample' or 'token'"
            )
        if eval_batch_size_unit is None:
            self.eval_batch_size_unit = self.batch_size_unit
        else:
            self.eval_batch_size_unit = str(eval_batch_size_unit).lower()
        if self.eval_batch_size_unit not in {"sample", "token"}:
            raise ValueError(
                f"Unsupported eval_batch_size_unit={eval_batch_size_unit}, expected 'sample' or 'token'"
            )

        super().__init__(
            model=model,
            args=args,
            callbacks=[self.metrics_callback],
            **kwargs,
        )

        # 清空 label_names，让 Transformers 在 eval 时使用 loss_without_labels 分支
        # 这样 prediction_step 会调用 compute_loss 而不是直接调用 model
        self.label_names = []

    def _build_bucket_batch_sampler(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool,
    ) -> DistributedLengthBucketBatchSampler:
        sampler = DistributedLengthBucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=shuffle,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
            bucket_multiplier=self.length_bucket_multiplier,
        )
        sampler.set_epoch(int(getattr(self.state, "epoch", 0) or 0))
        return sampler

    def _get_dataloader_with_batch_sampler(
        self,
        dataset,
        description: str,
        batch_sampler,
        dataloader_key: Optional[str] = None,
        is_training: bool = False,
    ):
        data_collator = self.data_collator
        if (
            datasets is not None
            and is_datasets_available()
            and isinstance(dataset, datasets.Dataset)
        ):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator,
                description=description,
            )

        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
        }

        if (
            self.args.dataloader_num_workers > 0
            and self.args.dataloader_prefetch_factor is not None
        ):
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )

        # 这里的 batch_sampler（Distributed*BucketBatchSampler）已经按 rank 分片过，
        # 如果再交给 accelerator.prepare()，Accelerate 可能会再做一次 shard，
        # 导致各 rank 迭代步数不一致，从而在 eval/save 等 collective 处出现 NCCL 死锁。
        #
        # 因此：自定义 distributed batch_sampler 路径下不做 prepare；Trainer 其余路径仍走基类逻辑。
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)

        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def _build_token_batch_sampler(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        shuffle: bool,
    ) -> DistributedTokenBucketBatchSampler:
        sampler = DistributedTokenBucketBatchSampler(
            lengths=lengths,
            max_tokens=max_tokens,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=shuffle,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
            bucket_multiplier=self.length_bucket_multiplier,
        )
        sampler.set_epoch(int(getattr(self.state, "epoch", 0) or 0))
        return sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.batch_size_unit == "token":
            if self.train_lengths is None:
                raise ValueError(
                    "Token-based batching requires precomputed train_lengths, but got None."
                )
            token_budget = int(getattr(self.args, "per_device_train_token_budget", 0))
            if token_budget <= 0:
                raise ValueError(
                    "per_device_train_token_budget must be positive in token batch mode."
                )
            batch_sampler = self._build_token_batch_sampler(
                lengths=self.train_lengths,
                max_tokens=token_budget,
                shuffle=True,
            )
            return self._get_dataloader_with_batch_sampler(
                dataset=self.train_dataset,
                description="Training",
                batch_sampler=batch_sampler,
                is_training=True,
            )

        if self.train_lengths is None:
            return super().get_train_dataloader()

        batch_sampler = self._build_bucket_batch_sampler(
            lengths=self.train_lengths,
            batch_size=self._train_batch_size,
            shuffle=True,
        )
        return self._get_dataloader_with_batch_sampler(
            dataset=self.train_dataset,
            description="Training",
            batch_sampler=batch_sampler,
            is_training=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # 回退到 sample-based eval 时，直接使用基类 DataLoader + 分布式 sampler，
        # 避免 length bucketing 在 DDP 下产生不一致的 batch 数。
        if self.eval_batch_size_unit == "sample":
            # 仅对 eval 走更保守的 DataLoader 配置：
            # - kernel/多进程 DataLoader/persistent_workers 组合在某些环境里会在 epoch 结束后 hang
            # - 训练侧的 bucketing/多 worker 保持不变
            orig_num_workers = self.args.dataloader_num_workers
            orig_persistent_workers = getattr(self.args, "dataloader_persistent_workers", False)
            orig_prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
            try:
                self.args.dataloader_num_workers = 0
                if hasattr(self.args, "dataloader_persistent_workers"):
                    self.args.dataloader_persistent_workers = False
                if hasattr(self.args, "dataloader_prefetch_factor"):
                    self.args.dataloader_prefetch_factor = None
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.args.dataloader_num_workers = orig_num_workers
                if hasattr(self.args, "dataloader_persistent_workers"):
                    self.args.dataloader_persistent_workers = orig_persistent_workers
                if hasattr(self.args, "dataloader_prefetch_factor"):
                    self.args.dataloader_prefetch_factor = orig_prefetch_factor

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        # 注意：训练可用 token batching，但 eval 可以回退到 sample batching 以降低 DDP remainder 不同步风险。
        if self.eval_batch_size_unit == "token":
            if self.eval_lengths is None:
                raise ValueError(
                    "Token-based batching requires precomputed eval_lengths, but got None."
                )
            token_budget = int(getattr(self.args, "per_device_eval_token_budget", 0))
            if token_budget <= 0:
                raise ValueError(
                    "per_device_eval_token_budget must be positive in token batch mode."
                )
            batch_sampler = self._build_token_batch_sampler(
                lengths=self.eval_lengths,
                max_tokens=token_budget,
                shuffle=False,
            )
            return self._get_dataloader_with_batch_sampler(
                dataset=eval_dataset,
                description="Evaluation",
                batch_sampler=batch_sampler,
                dataloader_key=dataloader_key,
            )

        if self.eval_lengths is None:
            return super().get_eval_dataloader(eval_dataset)

        batch_sampler = self._build_bucket_batch_sampler(
            lengths=self.eval_lengths,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
        )
        return self._get_dataloader_with_batch_sampler(
            dataset=eval_dataset,
            description="Evaluation",
            batch_sampler=batch_sampler,
            dataloader_key=dataloader_key,
        )

    def get_test_dataloader(self, test_dataset):
        # 与 eval 的 sample-based 回退保持一致，避免 prediction/test 路径在多 worker + persistent_workers 下 hang。
        if self.eval_batch_size_unit == "sample":
            orig_num_workers = self.args.dataloader_num_workers
            orig_persistent_workers = getattr(self.args, "dataloader_persistent_workers", False)
            orig_prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
            try:
                self.args.dataloader_num_workers = 0
                if hasattr(self.args, "dataloader_persistent_workers"):
                    self.args.dataloader_persistent_workers = False
                if hasattr(self.args, "dataloader_prefetch_factor"):
                    self.args.dataloader_prefetch_factor = None
                return super().get_test_dataloader(test_dataset)
            finally:
                self.args.dataloader_num_workers = orig_num_workers
                if hasattr(self.args, "dataloader_persistent_workers"):
                    self.args.dataloader_persistent_workers = orig_persistent_workers
                if hasattr(self.args, "dataloader_prefetch_factor"):
                    self.args.dataloader_prefetch_factor = orig_prefetch_factor

        return super().get_test_dataloader(test_dataset)

    def _load_scaler(self, checkpoint):
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is None:
            logger.warning(
                "Skipping GradScaler state restore because the current precision "
                "path does not use torch.cuda.amp.GradScaler."
            )
            return
        return super()._load_scaler(checkpoint)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        计算损失

        调用 model(..., return_loss=True)，模型负责计算 loss 和 metrics
        """
        # 如果 inputs 中已有 return_loss，使用它；否则显式设置
        if "return_loss" not in inputs:
            inputs["return_loss"] = True
        outputs = model(**inputs)

        # outputs 应该包含 "loss" 和 "metrics"
        loss = outputs["loss"]

        # 记录 metrics 到 callback（在 gather 之前提取）
        if self.metric_names:
            is_eval = inputs.get("is_eval", False)
            metrics = outputs.get("metrics")
            if metrics is not None:
                self.metrics_callback.update_metrics(
                    {"metrics": metrics}, is_training=not is_eval
                )

        if return_outputs:
            # 返回不含 metrics 的 outputs，避免 DataParallel gather 问题
            outputs_without_metrics = {k: v for k, v in outputs.items() if k != "metrics"}
            return loss, outputs_without_metrics
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        重写 prediction_step 以确保 eval 时也计算 loss 和 metrics

        基类的 prediction_step 只在 has_labels 或 can_return_loss 时才调用 compute_loss。
        由于我们的模型 return_loss 默认值是 False，基类不会调用 compute_loss，
        导致 eval 阶段没有 loss 和 metrics。

        这里我们显式传入 return_loss=True 来触发 loss 计算。
        """
        # 显式设置 return_loss=True
        inputs["return_loss"] = True
        inputs["is_eval"] = True
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
