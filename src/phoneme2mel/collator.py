"""
Phoneme2Mel Collator
处理变长序列的padding和批处理
"""
import torch


class Phoneme2MelCollator:
    """
    Collator for Phoneme2Mel dataset

    Pads sequences to same length within a batch
    """

    def __init__(self, pad_id=None):
        if pad_id is None:
            raise ValueError("pad_id must be provided explicitly to avoid colliding with real phoneme ids")
        self.pad_id = pad_id

    def __call__(self, batch):
        """
        Args:
            batch: list of dicts from Dataset.__getitem__

        Returns:
            dict with batched tensors
        """
        B = len(batch)

        # Extract lengths
        phoneme_lens = [int(item['phoneme_len']) for item in batch]
        mel_lens = [int(item['mel_len']) for item in batch]
        downsampled_phoneme_lens = [(length + 1) // 2 for length in phoneme_lens]

        max_phoneme_len = max(phoneme_lens)
        max_mel_len = max(mel_lens)

        # Pad phoneme_ids
        phoneme_ids_padded = torch.full((B, max_phoneme_len), self.pad_id, dtype=torch.long)
        src_key_padding_mask = torch.ones((B, max_phoneme_len), dtype=torch.bool)  # True = masked

        for i, item in enumerate(batch):
            L = item['phoneme_ids'].size(0)
            phoneme_ids_padded[i, :L] = item['phoneme_ids']
            src_key_padding_mask[i, :L] = False  # False = not masked

        # Pad mel
        mel_dim = batch[0]['mel'].size(1)
        mel_padded = torch.zeros((B, max_mel_len, mel_dim))
        mel_mask = torch.zeros((B, max_mel_len), dtype=torch.bool)  # True = valid

        for i, item in enumerate(batch):
            L = item['mel'].size(0)
            mel_padded[i, :L] = item['mel']
            mel_mask[i, :L] = True  # Valid frames

        # Stack xvectors
        xvector = torch.stack([item['xvector'] for item in batch])

        # Collect metadata
        wav_ids = [item['wav_id'] for item in batch]
        spk_ids = [item['spk_id'] for item in batch]

        return {
            'phoneme_ids': phoneme_ids_padded,  # [B, T_phoneme]
            'xvector': xvector,  # [B, xvector_dim]
            'src_key_padding_mask': src_key_padding_mask,  # [B, T_phoneme]
            'mel': mel_padded,  # [B, T_mel, mel_dim]
            'mel_mask': mel_mask,  # [B, T_mel]
            'phoneme_lens': phoneme_lens,
            'downsampled_phoneme_lens': torch.tensor(downsampled_phoneme_lens, dtype=torch.long),
            'mel_lens': mel_lens,
            'wav_ids': wav_ids,
            'spk_ids': spk_ids,
        }
