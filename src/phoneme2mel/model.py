"""
RoPE-based phoneme2mel model aligned with phoneme2token backbone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from src.phoneme2mel.config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.outer(positions.float(), inv_freq.float())
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=q.dtype, device=q.device)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=q.dtype, device=q.device)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


@dataclass
class Phoneme2MelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    mel_pred: Optional[torch.Tensor] = None
    mel_pred_before: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None


class TacotronPostNet(nn.Module):
    """Tacotron-style PostNet."""

    def __init__(
        self,
        mel_dim: int = 128,
        n_layers: int = 5,
        dim: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        )
        for _ in range(n_layers - 2):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(dim, mel_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(dropout),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv in self.conv_layers[:-1]:
            x = conv(x)
        x = self.conv_layers[-1](x)
        return x.transpose(1, 2)


class LegacyPostNet(nn.Module):
    """PostNet matching src.phoneme2mel_legacy.model.Postnet exactly."""

    def __init__(
        self,
        idim: int,
        odim: int,
        n_layers: int = 5,
        n_chans: int = 256,
        kernel_size: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_chans = idim
        for i in range(n_layers):
            out_chans = odim if i == n_layers - 1 else n_chans
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_chans,
                        out_chans,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_chans),
                    nn.Tanh() if i != n_layers - 1 else nn.Identity(),
                    nn.Dropout(dropout_rate),
                )
            )
            in_chans = out_chans
        self.postnet = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.postnet:
            x = layer(x)
        return x.transpose(1, 2)


class ScaledPositionalEncoding(nn.Module):
    """Legacy absolute sinusoidal position encoding with a learned scale."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, length: int, device=None) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.size(1)
        if self.pe.size(0) < seq_len or self.pe.device != hidden.device:
            self.pe = self._build_pe(max(self.pe.size(0), seq_len), device=hidden.device)
        return hidden + self.alpha * self.pe[:seq_len].to(hidden.dtype).unsqueeze(0)


class LegacyConditioningEncoderLayer(nn.Module):
    """Legacy Post-Norm encoder layer with per-layer speaker conditioning."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead // 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.cond_proj = nn.Linear(config.d_model, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = F.relu if config.activation == "relu" else F.gelu

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.self_attn(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        hidden = self.norm1(hidden + self.dropout1(residual))

        condition = self.cond_proj(condition).unsqueeze(1)
        condition = condition.expand(-1, hidden.size(1), -1)
        residual = self.cross_attn(
            hidden,
            condition,
            condition,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        hidden = self.norm2(hidden + self.dropout2(residual))

        residual = self.linear2(self.dropout(self.activation(self.linear1(hidden))))
        return self.norm3(hidden + self.dropout3(residual))


class LegacyConditioningEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [LegacyConditioningEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(
                hidden,
                condition=condition,
                key_padding_mask=key_padding_mask,
            )
        return hidden


class RotarySelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = float(config.dropout)
        self.register_buffer("inv_freq", self._build_inv_freq(), persistent=False)

    def _build_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return 1.0 / (
            10000
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )

    def _ensure_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = self.inv_freq
        expected_size = self.head_dim // 2
        needs_reset = (
            inv_freq.numel() != expected_size
            or not torch.isfinite(inv_freq).all()
            or bool(inv_freq.le(0).any())
            or bool(inv_freq.gt(1.0).any())
        )
        if needs_reset:
            inv_freq = self._build_inv_freq(device=device)
            self.inv_freq = inv_freq
        elif inv_freq.device != device:
            inv_freq = inv_freq.to(device=device)
            self.inv_freq = inv_freq
        return inv_freq

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        qkv = self.qkv_proj(hidden)
        qkv = qkv.view(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q, k = _apply_rope(q, k, positions, self._ensure_inv_freq(hidden.device))

        blocked = attn_mask.unsqueeze(0).unsqueeze(0) | key_padding_mask.unsqueeze(1).unsqueeze(2)
        additive_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            device=hidden.device,
            dtype=q.dtype,
        )
        additive_mask = additive_mask.masked_fill(blocked, torch.finfo(q.dtype).min)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn)


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = RotarySelfAttention(config)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden)
        hidden = hidden + self.dropout1(
            self.self_attn(
                attn_input,
                positions=positions,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )
        ff_input = self.norm2(hidden)
        ff_hidden = self.linear2(self.dropout(F.gelu(self.linear1(ff_input))))
        return hidden + self.dropout2(ff_hidden)


class RotaryTransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [RotaryTransformerEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(
                hidden,
                positions=positions,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return hidden


class Phoneme2MelModel(PreTrainedModel):
    """Phoneme-conditioned mel predictor with RoPE backbone and PostNet."""

    config_class = ModelConfig
    base_model_prefix = "phoneme2mel"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.xvector_dim = config.xvector_dim
        self.d_model = config.d_model
        self.num_phonemes = config.num_phonemes
        self.vocab_size = config.vocab_size
        self.pad_id = config.pad_id
        self.mel_dim = config.mel_dim

        self.xvector_proj = nn.Linear(config.xvector_dim, config.d_model)
        self.phoneme_embed = nn.Embedding(
            config.vocab_size,
            config.phoneme_embed_dim,
            padding_idx=config.pad_id,
        )
        self.phoneme_downsample = nn.Conv1d(
            config.phoneme_embed_dim,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        if config.backbone_type == "legacy":
            self.positional_encoding = ScaledPositionalEncoding(
                config.d_model,
                max_len=config.max_position_embeddings,
            )
            self.transformer = LegacyConditioningEncoder(config)
            self.final_norm = nn.Identity()
        else:
            self.positional_encoding = nn.Identity()
            self.transformer = RotaryTransformerEncoder(config)
            self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mel_pred_linear = nn.Linear(config.d_model, config.mel_dim)
        if config.postnet_type == "legacy":
            self.postnet = LegacyPostNet(
                idim=config.mel_dim,
                odim=config.mel_dim,
                n_layers=config.postnet_n_layers,
                n_chans=config.postnet_dim,
                kernel_size=config.postnet_kernel_size,
                dropout_rate=config.postnet_dropout,
            )
        else:
            self.postnet = TacotronPostNet(
                mel_dim=config.mel_dim,
                n_layers=config.postnet_n_layers,
                dim=config.postnet_dim,
                kernel_size=config.postnet_kernel_size,
                dropout=config.postnet_dropout,
            )
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            module.weight[module.padding_idx].zero_()
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_xvector(self, xvector: torch.Tensor) -> None:
        if xvector.dim() != 2:
            raise ValueError(
                f"xvector must have shape [B, {self.config.xvector_dim}], got {tuple(xvector.shape)}"
            )
        if xvector.size(-1) != self.config.xvector_dim:
            raise ValueError(
                f"xvector dim mismatch: expected {self.config.xvector_dim}, got {xvector.size(-1)}"
            )

    @staticmethod
    def _build_bidirectional_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)

    @staticmethod
    def _downsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def _downsample_phoneme_embeddings(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_key_padding_mask: Optional[torch.Tensor] = None,
        downsampled_phoneme_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, phoneme_len = phoneme_ids.shape
        phoneme_emb = self.phoneme_embed(phoneme_ids)
        downsampled = self.phoneme_downsample(phoneme_emb.transpose(1, 2)).transpose(1, 2)

        if downsampled_phoneme_lens is not None:
            valid_lens = downsampled_phoneme_lens.to(device=phoneme_ids.device, dtype=torch.long)
        elif phoneme_key_padding_mask is None:
            valid_lens = torch.full(
                (batch_size,),
                (phoneme_len + 1) // 2,
                device=phoneme_ids.device,
                dtype=torch.long,
            )
        else:
            phoneme_lens = phoneme_key_padding_mask.logical_not().sum(dim=1)
            valid_lens = self._downsample_lengths(phoneme_lens)

        valid_lens = valid_lens.clamp(min=0, max=downsampled.size(1))
        positions = torch.arange(downsampled.size(1), device=phoneme_ids.device).unsqueeze(0)
        downsampled_mask = positions >= valid_lens.unsqueeze(1)
        return downsampled, downsampled_mask

    def _forward(
        self,
        phoneme_ids: torch.Tensor,
        xvector: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        downsampled_phoneme_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_xvector(xvector)
        batch_size = phoneme_ids.size(0)

        xvector_hidden = self.xvector_proj(xvector).unsqueeze(1)
        phoneme_hidden, phoneme_hidden_mask = self._downsample_phoneme_embeddings(
            phoneme_ids=phoneme_ids,
            phoneme_key_padding_mask=src_key_padding_mask,
            downsampled_phoneme_lens=downsampled_phoneme_lens,
        )
        hidden = torch.cat([xvector_hidden, phoneme_hidden], dim=1)

        seq_len = hidden.size(1)
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        positions = torch.arange(seq_len, device=hidden.device)
        xvector_mask = torch.zeros((batch_size, 1), device=hidden.device, dtype=torch.bool)
        key_padding_mask = torch.cat([xvector_mask, phoneme_hidden_mask], dim=1)
        attn_mask = self._build_bidirectional_attention_mask(seq_len=seq_len, device=hidden.device)

        hidden = self.positional_encoding(hidden)
        if self.config.backbone_type == "legacy":
            hidden = self.transformer(
                hidden,
                condition=xvector_hidden.squeeze(1),
                key_padding_mask=key_padding_mask,
            )
        else:
            hidden = self.transformer(
                hidden,
                positions=positions,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        hidden = self.final_norm(hidden)
        phoneme_only_hidden = hidden[:, 1:, :]

        mel_pred_before = self.mel_pred_linear(phoneme_only_hidden)
        mel_pred = self.postnet(mel_pred_before) + mel_pred_before
        return mel_pred, mel_pred_before

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        xvector: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        mel_mask: Optional[torch.Tensor] = None,
        downsampled_phoneme_lens: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Phoneme2MelOutput]:
        mel_pred, mel_pred_before = self._forward(
            phoneme_ids=phoneme_ids,
            xvector=xvector,
            src_key_padding_mask=src_key_padding_mask,
            downsampled_phoneme_lens=downsampled_phoneme_lens,
        )

        if not return_loss:
            return mel_pred, mel_pred_before

        if mel is None:
            raise ValueError("mel is required when return_loss=True")

        from .model_metrics import compute_metrics

        metrics_dict = compute_metrics(
            mel_pred=mel_pred,
            mel_pred_before=mel_pred_before,
            mel_target=mel,
            mel_mask=mel_mask,
        )

        return Phoneme2MelOutput(
            loss=metrics_dict["loss"],
            mel_pred=mel_pred,
            mel_pred_before=mel_pred_before,
            metrics={
                "loss_after_postnet": metrics_dict["loss_after_postnet"].item(),
                "loss_before_postnet": metrics_dict["loss_before_postnet"].item(),
            },
        )

    def save_pretrained(self, output_dir, **kwargs):
        self.config.save_pretrained(output_dir)
        super().save_pretrained(output_dir, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
