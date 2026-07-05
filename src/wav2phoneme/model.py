"""
模型定义 - Wav2PhonemeModel
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import ModelOutput

from src.wav2phoneme.config import ModelConfig
from src.wav2phoneme.loss import forced_alignment_loss, ctc_loss, interpolate_loss


@dataclass
class Wav2PhonemeOutput(ModelOutput):
    """Wav2Phoneme 模型输出"""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    metrics: Optional[Dict[str, float]] = None


class Wav2PhonemeModel(PreTrainedModel):
    """
    Wav2Phoneme 模型
    
    架构:
        Wav2Vec2Model (pretrained, 完全不冻结)
            ↓ [hidden_size, ~50fps]
        Linear(hidden_size → num_phonemes)
            ↓ [num_phonemes, ~50fps]
    """
    
    config_class = ModelConfig
    base_model_prefix = "wav2phoneme"
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        
        # 外层 HF from_pretrained 会在 meta device 上先构建空模型，此时不能在内部再次
        # 调用 from_pretrained。普通训练初始化仍然保留 base wav2vec2 预训练权重。
        if torch.empty(0).device.type == "meta":
            wav2vec_config = Wav2Vec2Config.from_pretrained(config.wav2vec_model_path)
            self.wav2vec = Wav2Vec2Model(wav2vec_config)
        else:
            self.wav2vec = Wav2Vec2Model.from_pretrained(config.wav2vec_model_path)
        
        # 获取 hidden_size
        hidden_size = self.wav2vec.config.hidden_size
        
        # Linear 投影层
        self.classifier = nn.Linear(hidden_size, config.num_phonemes)
        
        # 初始化 classifier
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # 保存配置
        self.num_phonemes = config.num_phonemes
        self.mode = config.mode
        self.ctc_blank_id = int(getattr(config, "ctc_blank_id", 0))

    def _get_encoder_output_lengths(self, attention_mask: torch.Tensor, max_length: int) -> torch.Tensor:
        audio_lengths = attention_mask.sum(dim=1)
        if hasattr(self.wav2vec, "_get_feat_extract_output_lengths"):
            lengths = self.wav2vec._get_feat_extract_output_lengths(audio_lengths)
        else:
            lengths = torch.full_like(audio_lengths, max_length)
        return lengths.clamp(max=max_length).to(dtype=torch.long)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Wav2PhonemeOutput]:
        """
        Forward pass

        Args:
            input_values: [B, T], 音频波形
            attention_mask: [B, T], 注意力掩码
            labels: [B, T'], 标签（训练时）
            label_mask: [B, T'], 标签掩码（forced_alignment/interpolate 模式）
            label_lengths: [B], 标签长度（CTC 模式）
            return_loss: 是否计算损失和指标

        Returns:
            若 return_loss=False:
                logits
            若 return_loss=True:
                Wav2PhonemeOutput(loss=..., logits=..., metrics=...)
        """
        # wav2vec forward
        outputs = self.wav2vec(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # hidden_states: [B, T', hidden_size]
        hidden_states = outputs.last_hidden_state

        # classifier: [B, T', num_phonemes]
        logits = self.classifier(hidden_states)

        if not return_loss:
            return logits

        # 计算 loss 和 metrics
        if labels is None:
            return Wav2PhonemeOutput(loss=None, logits=logits, metrics=None)

        # 计算 loss
        if self.mode == "forced_alignment":
            loss = forced_alignment_loss(logits, labels, label_mask)
        elif self.mode == "ctc":
            if attention_mask is None:
                raise ValueError("attention_mask is required for CTC mode")
            if label_lengths is None:
                raise ValueError("label_lengths is required for CTC mode")
            input_lengths = self._get_encoder_output_lengths(attention_mask, logits.shape[1])
            loss = ctc_loss(
                logits,
                labels,
                input_lengths=input_lengths,
                label_lengths=label_lengths,
                blank=self.ctc_blank_id,
            )
        elif self.mode == "interpolate":
            loss = interpolate_loss(logits, labels, label_mask)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 计算 metrics
        from .model_metrics import compute_metrics

        metrics = compute_metrics(
            logits=logits,
            labels=labels,
            label_mask=label_mask,
            attention_mask=attention_mask,
            mode=self.mode,
            input_lengths=input_lengths if self.mode == "ctc" else None,
            label_lengths=label_lengths if self.mode == "ctc" else None,
        )

        return Wav2PhonemeOutput(
            loss=loss,
            logits=logits,
            metrics=metrics,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> List[torch.Tensor]:
        """
        推理：生成音素 ID 序列
        
        Args:
            input_values: [B, T], 音频波形
            attention_mask: [B, T], 注意力掩码
            return_logits: 是否返回 logits
        
        Returns:
            List of [T'], 每个 sample 的音素 ID 序列
            如果 return_logits=True，返回 (phoneme_ids_list, logits)
        """
        self.eval()
        
        # Forward
        outputs = self.wav2vec(
            input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        
        # 取 argmax
        phoneme_ids = logits.argmax(dim=-1)  # [B, T']
        
        # 转换为列表
        phoneme_ids_list = []
        for i in range(phoneme_ids.shape[0]):
            # 根据 attention_mask 裁剪
            if attention_mask is not None:
                # wav2vec 的输出长度可能与输入不同
                # 这里简单处理
                length = phoneme_ids.shape[1]
            else:
                length = phoneme_ids.shape[1]
            phoneme_ids_list.append(phoneme_ids[i, :length])
        
        if return_logits:
            return phoneme_ids_list, logits
        return phoneme_ids_list
    
    def save_pretrained(self, output_dir, **kwargs):
        """保存模型"""
        # 保存 config
        self.config.save_pretrained(output_dir)
        # 保存模型权重
        super().save_pretrained(output_dir, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """从 checkpoint 或 config 加载模型"""
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
