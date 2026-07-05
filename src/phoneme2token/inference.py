"""Inference helpers for phoneme2token."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from src.phoneme2token.dataset import load_phone_dict, load_phoneme_names
from src.phoneme2token.model import Phoneme2TokenModel


class Phoneme2TokenInference:
    def __init__(
        self,
        checkpoint_path: str,
        phone_dict_path: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = Phoneme2TokenModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.phone_to_id = load_phone_dict(Path(phone_dict_path))
        self.sil_id = self.phone_to_id.get("sil", 0)

    def phoneme_names_to_ids(self, phoneme_names: Iterable[str], already_50hz: bool = True) -> list[int]:
        ids = [self.phone_to_id.get(name, self.sil_id) for name in phoneme_names]
        if not already_50hz:
            ids = ids[0::2]
        return ids

    def infer(
        self,
        phoneme_ids: Iterable[int],
        xvector: np.ndarray,
        target_len: Optional[int] = None,
    ) -> dict:
        phoneme_tensor = torch.tensor(list(phoneme_ids), dtype=torch.long, device=self.device)
        xvector_tensor = torch.tensor(xvector, dtype=torch.float32, device=self.device)
        if target_len is None:
            target_len = round(phoneme_tensor.numel() / 2) + self.model.config.length_pad_margin
        generated = self.model.generate_tokens(
            phoneme_ids=phoneme_tensor,
            xvector=xvector_tensor,
            target_len=target_len,
        )
        raw_tokens = generated.detach().cpu().numpy().astype(np.int64)
        tokens = raw_tokens[raw_tokens != self.model.config.token_pad_id]
        tokens = tokens[(tokens >= 0) & (tokens < self.model.config.speech_vocab_size)]
        return {
            "tokens": tokens.tolist(),
            "raw_tokens": raw_tokens.tolist(),
            "target_len": int(target_len),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phoneme2token inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phone-dict", required=True)
    parser.add_argument("--phone-file", required=True)
    parser.add_argument("--xvector-path", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-rate", choices=["50hz", "100hz"], default="50hz")
    args = parser.parse_args()

    inferencer = Phoneme2TokenInference(args.checkpoint, args.phone_dict, args.device)
    names = load_phoneme_names(Path(args.phone_file))
    phoneme_ids = inferencer.phoneme_names_to_ids(names, already_50hz=args.input_rate == "50hz")
    result = inferencer.infer(phoneme_ids, np.load(args.xvector_path))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(result["tokens"], dtype=np.int64))
        output_path.with_suffix(".json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

