"""
Phoneme utilities
"""
import json
from pathlib import Path


def load_phone_dict(phone_dict_path):
    """Load phone dictionary from JSON file."""
    with open(phone_dict_path, 'r', encoding='utf-8') as f:
        phone_dict = json.load(f)
    return phone_dict


def save_phone_dict(phone_dict, phone_dict_path):
    """Save phone dictionary to JSON file."""
    with open(phone_dict_path, 'w', encoding='utf-8') as f:
        json.dump(phone_dict, f, ensure_ascii=False, indent=2)


def deduplicate_phoneme(phoneme_ids):
    """
    Remove consecutive duplicates from phoneme sequence.
    E.g., [1, 1, 1, 2, 2, 3, 3, 3, 1, 1] -> [1, 2, 3, 1]
    """
    if len(phoneme_ids) == 0:
        return []

    deduped = [phoneme_ids[0]]
    for p in phoneme_ids[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    return deduped


def remove_sil(phoneme_ids, sil_id=0):
    """
    Remove silence tokens from phoneme sequence.
    """
    return [p for p in phoneme_ids if p != sil_id]


def downsample_phoneme(phoneme_ids, factor=2):
    """
    Downsample phoneme sequence by factor (using mean pooling).
    """
    L = len(phoneme_ids)
    if L % factor != 0:
        # Truncate to make it divisible
        L = (L // factor) * factor
        phoneme_ids = phoneme_ids[:L]

    phoneme_ids = phoneme_ids.reshape(-1, factor)
    downsampled = phoneme_ids.mean(axis=1).astype(phoneme_ids.dtype)
    return downsampled.tolist()
