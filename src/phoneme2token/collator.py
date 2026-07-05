"""Batch collation for phoneme2token."""
import torch
from torch.nn.utils.rnn import pad_sequence

from .frontend import load_audio_mono_16k, pad_audio_batch


class Phoneme2TokenCollator:
    def __init__(
        self,
        phoneme_pad_id: int,
        token_pad_id: int,
        ignore_index: int = -100,
    ):
        self.phoneme_pad_id = int(phoneme_pad_id)
        self.token_pad_id = int(token_pad_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: list[dict]) -> dict:
        if "waveform" in batch[0] or "wav_path" in batch[0]:
            if "waveform" in batch[0]:
                waveforms = [item["waveform"] for item in batch]
            else:
                waveforms = [load_audio_mono_16k(item["wav_path"]) for item in batch]
            input_values, attention_mask = pad_audio_batch(waveforms)
            decoder_input_tokens = pad_sequence(
                [item["decoder_input_tokens"] for item in batch],
                batch_first=True,
                padding_value=self.token_pad_id,
            )
            labels = pad_sequence(
                [item["labels"] for item in batch],
                batch_first=True,
                padding_value=self.ignore_index,
            )
            target_key_padding_mask = labels.eq(self.ignore_index)
            return {
                "input_values": input_values,
                "attention_mask": attention_mask,
                "decoder_input_tokens": decoder_input_tokens,
                "target_key_padding_mask": target_key_padding_mask,
                "labels": labels,
                "xvector": torch.stack([item["xvector"] for item in batch]),
                "phoneme_lens": [item["phoneme_len"] for item in batch],
                "target_lens": [item["target_len"] for item in batch],
                "token_lens": [item["token_len"] for item in batch],
                "base_lens": [item["base_len"] for item in batch],
                "wav_ids": [item["wav_id"] for item in batch],
                "spk_ids": [item["spk_id"] for item in batch],
                "dataset_names": [item["dataset_name"] for item in batch],
                "wav_paths": [item["wav_path"] for item in batch],
            }

        phoneme_ids = pad_sequence(
            [item["phoneme_ids"] for item in batch],
            batch_first=True,
            padding_value=self.phoneme_pad_id,
        )
        phoneme_key_padding_mask = phoneme_ids.eq(self.phoneme_pad_id)
        decoder_input_tokens = pad_sequence(
            [item["decoder_input_tokens"] for item in batch],
            batch_first=True,
            padding_value=self.token_pad_id,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=self.ignore_index,
        )
        target_key_padding_mask = labels.eq(self.ignore_index)

        return {
            "phoneme_ids": phoneme_ids,
            "phoneme_key_padding_mask": phoneme_key_padding_mask,
            "decoder_input_tokens": decoder_input_tokens,
            "target_key_padding_mask": target_key_padding_mask,
            "labels": labels,
            "xvector": torch.stack([item["xvector"] for item in batch]),
            "phoneme_lens": [item["phoneme_len"] for item in batch],
            "target_lens": [item["target_len"] for item in batch],
            "token_lens": [item["token_len"] for item in batch],
            "base_lens": [item["base_len"] for item in batch],
            "wav_ids": [item["wav_id"] for item in batch],
            "spk_ids": [item["spk_id"] for item in batch],
            "dataset_names": [item["dataset_name"] for item in batch],
        }
