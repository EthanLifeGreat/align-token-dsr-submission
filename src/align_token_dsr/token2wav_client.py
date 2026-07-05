"""Client for an external frozen CosyVoice2 Token2Wav service."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np


def _read_prompt(path: Path | None) -> str | None:
    if path is None:
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def synthesize_tokens(
    token_path: Path,
    output_wav: Path,
    service_url: str,
    prompt_wav: Path | None = None,
    prompt_text: str = "",
    timeout: float = 120.0,
) -> None:
    token_ids = np.load(token_path).astype(np.int64).reshape(-1).tolist()
    payload: dict[str, Any] = {
        "token_ids": token_ids,
        "prompt_text": prompt_text,
    }
    prompt_wav_base64 = _read_prompt(prompt_wav)
    if prompt_wav_base64 is not None:
        payload["prompt_wav_base64"] = prompt_wav_base64

    request = Request(
        service_url.rstrip("/") + "/token2wav",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    wav_base64 = result.get("wav_base64")
    if not wav_base64:
        raise RuntimeError(f"Token2Wav response did not contain wav_base64: {result}")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    output_wav.write_bytes(base64.b64decode(wav_base64))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render semantic tokens with an external CosyVoice2 Token2Wav service.")
    parser.add_argument("--token-npy", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt-wav", type=Path, default=None)
    parser.add_argument("--prompt-text", default="")
    args = parser.parse_args()
    synthesize_tokens(
        token_path=args.token_npy,
        output_wav=args.output_wav,
        service_url=args.service_url,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
    )


if __name__ == "__main__":
    main()

