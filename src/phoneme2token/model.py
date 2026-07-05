"""Transformer models for phoneme to CosyVoice2 token prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from src.phoneme2token.config import ModelConfig
from src.phoneme2token.frontend import FrameArgmaxFrontend
from src.phoneme2token.model_metrics import compute_metrics


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


@dataclass
class Phoneme2TokenOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None


class Phoneme2TokenModel(PreTrainedModel):
    """Phoneme-conditioned semantic token predictor."""

    config_class = ModelConfig
    base_model_prefix = "phoneme2token"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.__dict__["_frameargmax_frontend_helper"] = None
        self.xvector_proj = nn.Linear(config.xvector_dim, config.d_model)
        self.phoneme_embed = nn.Embedding(
            config.phoneme_vocab_size,
            config.d_model,
            padding_idx=config.phoneme_pad_id,
        )
        self.token_embed = nn.Embedding(
            config.token_input_vocab_size,
            config.d_model,
        )
        self.transformer = RotaryTransformerEncoder(config)
        self.final_norm = nn.LayerNorm(config.d_model)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.token_output_vocab_size)
            self.lm_head_bias = None
        else:
            self.lm_head = None
            self.lm_head_bias = nn.Parameter(torch.zeros(config.token_output_vocab_size))
        self.post_init()

    def _get_frameargmax_frontend_helper(self) -> FrameArgmaxFrontend:
        helper = self.__dict__.get("_frameargmax_frontend_helper")
        if helper is None:
            if self.config.frontend_mode != "wav2phoneme_ctc_frameargmax":
                raise RuntimeError("frontend helper requested in precomputed mode")
            helper = FrameArgmaxFrontend(
                wav2phoneme_ckpt=str(self.config.wav2phoneme_ckpt),
                phone_dict_path=str(self.config.phone_dict_path),
                device=str(self.device),
                cache_dir=self.config.frontend_cache_dir,
                repeat_factor=self.config.frontend_repeat_factor,
                blank_id=self.config.frontend_blank_id,
            )
            self.__dict__["_frameargmax_frontend_helper"] = helper
        return helper

    def _extract_frontend_batch(
        self,
        *,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        dataset_names: list[str],
        spk_ids: list[str],
        wav_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        helper = self._get_frameargmax_frontend_helper()
        frontend_batch = helper.extract_batch_tensors(
            input_values=input_values,
            attention_mask=attention_mask,
            dataset_names=dataset_names,
            spk_ids=spk_ids,
            wav_ids=wav_ids,
            pad_id=self.config.phoneme_pad_id,
            output_device=input_values.device,
        )
        if not frontend_batch.all_identity_pass:
            bad = [
                f"{dataset_names[index]}/{spk_ids[index]}/{wav_ids[index]}"
                for index, is_valid in enumerate(frontend_batch.identity_pass_flags)
                if not is_valid
            ]
            raise RuntimeError(f"2x upsample identity check failed for: {bad[:8]}")
        return (
            frontend_batch.phoneme_ids,
            frontend_batch.phoneme_key_padding_mask,
            frontend_batch.metrics,
        )

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

    def _compute_logits(self, token_hidden: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(token_hidden)
        output_weight = self.token_embed.weight[: self.config.token_output_vocab_size]
        return F.linear(token_hidden, output_weight, self.lm_head_bias)

    def _build_attention_mask(
        self,
        prefix_len: int,
        target_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_len = prefix_len + target_len
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)

        # Prefix tokens can see the full prefix but not speech/PAD positions.
        if target_len > 0:
            mask[:prefix_len, prefix_len:] = True

        # Speech/PAD positions can see the full prefix and only previous target positions.
        future = torch.triu(
            torch.ones((target_len, target_len), device=device, dtype=torch.bool),
            diagonal=1,
        )
        mask[prefix_len:, prefix_len:] = future
        return mask

    def _forward_decoder_only_logits(
        self,
        phoneme_ids: torch.Tensor,
        xvector: torch.Tensor,
        decoder_input_tokens: torch.Tensor,
        phoneme_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, phoneme_len = phoneme_ids.shape
        target_len = decoder_input_tokens.size(1)

        xvector_emb = self.xvector_proj(xvector).unsqueeze(1)
        phoneme_emb = self.phoneme_embed(phoneme_ids)
        token_emb = self.token_embed(decoder_input_tokens)
        hidden = torch.cat([xvector_emb, phoneme_emb, token_emb], dim=1)

        seq_len = hidden.size(1)
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        positions = torch.arange(seq_len, device=hidden.device)

        if phoneme_key_padding_mask is None:
            phoneme_key_padding_mask = torch.zeros(
                (batch_size, phoneme_len), device=hidden.device, dtype=torch.bool
            )
        if target_key_padding_mask is None:
            target_key_padding_mask = torch.zeros(
                (batch_size, target_len), device=hidden.device, dtype=torch.bool
            )
        xvector_mask = torch.zeros((batch_size, 1), device=hidden.device, dtype=torch.bool)
        key_padding_mask = torch.cat(
            [xvector_mask, phoneme_key_padding_mask, target_key_padding_mask],
            dim=1,
        )
        attn_mask = self._build_attention_mask(
            prefix_len=1 + phoneme_len,
            target_len=target_len,
            device=hidden.device,
        )

        hidden = self.transformer(
            hidden,
            positions=positions,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        hidden = self.final_norm(hidden)
        token_hidden = hidden[:, 1 + phoneme_len :, :]
        return self._compute_logits(token_hidden)

    def _forward_logits(
        self,
        phoneme_ids: torch.Tensor,
        xvector: torch.Tensor,
        decoder_input_tokens: torch.Tensor,
        phoneme_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_decoder_only_logits(
            phoneme_ids=phoneme_ids,
            xvector=xvector,
            decoder_input_tokens=decoder_input_tokens,
            phoneme_key_padding_mask=phoneme_key_padding_mask,
            target_key_padding_mask=target_key_padding_mask,
        )

    def forward(
        self,
        xvector: torch.Tensor,
        decoder_input_tokens: torch.Tensor,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dataset_names: Optional[list[str]] = None,
        spk_ids: Optional[list[str]] = None,
        wav_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Phoneme2TokenOutput]:
        frontend_metrics: dict[str, float] = {}
        if phoneme_ids is None:
            if self.config.frontend_mode != "wav2phoneme_ctc_frameargmax":
                raise ValueError("phoneme_ids are required in precomputed frontend mode")
            if input_values is None or attention_mask is None:
                raise ValueError("input_values and attention_mask are required for frameargmax mode")
            if dataset_names is None or spk_ids is None or wav_ids is None:
                raise ValueError("dataset_names, spk_ids, and wav_ids are required for frameargmax mode")
            phoneme_ids, phoneme_key_padding_mask, frontend_metrics = self._extract_frontend_batch(
                input_values=input_values,
                attention_mask=attention_mask,
                dataset_names=dataset_names,
                spk_ids=spk_ids,
                wav_ids=wav_ids,
            )

        logits = self._forward_logits(
            phoneme_ids=phoneme_ids,
            xvector=xvector,
            decoder_input_tokens=decoder_input_tokens,
            phoneme_key_padding_mask=phoneme_key_padding_mask,
            target_key_padding_mask=target_key_padding_mask,
        )
        if not return_loss:
            return logits

        if labels is None:
            raise ValueError("labels are required when return_loss=True")
        loss_all = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss = loss_all
        metrics = compute_metrics(
            logits=logits.detach(),
            labels=labels.detach(),
            token_pad_id=self.config.token_pad_id,
            ignore_index=-100,
        )
        metrics.update(
            {
                "loss_all": loss_all.detach().item(),
            }
        )
        metrics.update(frontend_metrics)
        metrics = {
            name: torch.tensor(value, device=logits.device, dtype=torch.float32)
            for name, value in metrics.items()
        }
        return Phoneme2TokenOutput(loss=loss, logits=logits, metrics=metrics)

    @torch.no_grad()
    def generate_tokens(
        self,
        phoneme_ids: torch.Tensor,
        xvector: torch.Tensor,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy fixed-length generation. Returns tokens with PAD stripped per caller."""
        self.eval()
        if phoneme_ids.dim() == 1:
            phoneme_ids = phoneme_ids.unsqueeze(0)
        if xvector.dim() == 1:
            xvector = xvector.unsqueeze(0)
        if phoneme_ids.size(0) != 1:
            raise ValueError("generate_tokens currently supports batch size 1")

        if target_len is None:
            target_len = round(phoneme_ids.size(1) / 2) + self.config.length_pad_margin
        if target_len <= 0:
            return torch.empty((0,), device=phoneme_ids.device, dtype=torch.long)

        generated: list[torch.Tensor] = []
        decoder_input = torch.full(
            (1, 1),
            self.config.bos_id,
            device=phoneme_ids.device,
            dtype=torch.long,
        )
        phoneme_mask = torch.zeros_like(phoneme_ids, dtype=torch.bool)

        for _ in range(target_len):
            target_mask = torch.zeros_like(decoder_input, dtype=torch.bool)
            logits = self._forward_logits(
                phoneme_ids=phoneme_ids,
                xvector=xvector,
                decoder_input_tokens=decoder_input,
                phoneme_key_padding_mask=phoneme_mask,
                target_key_padding_mask=target_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

        return torch.cat(generated, dim=0)
