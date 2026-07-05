from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    collect_scp_keys,
    get_default_parts_dir,
    get_default_token_dir,
    resolve_project_root,
)


def iter_part_paths(parts_dir: Path) -> list[Path]:
    paths = sorted(path for path in parts_dir.glob("part_*_of_*") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No part_*_of_* files found in {parts_dir}")
    return paths


def materialize_parts(
    parts_dir: Path,
    token_dir: Path,
    expected_scp: Path | None = None,
) -> dict[str, object]:
    expected_keys = collect_scp_keys(expected_scp) if expected_scp else None
    seen_keys: set[str] = set()
    written = 0
    duplicate_same_count = 0
    token_dir.mkdir(parents=True, exist_ok=True)

    for part_path in iter_part_paths(parts_dir):
        with part_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {part_path}:{line_number}") from exc

                key = record.get("key")
                code = record.get("code")
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Invalid key in {part_path}:{line_number}")

                array = np.asarray(code, dtype=np.int32)
                if array.ndim != 1:
                    raise ValueError(f"Non-1D token array for key {key} in {part_path}:{line_number}")
                if array.size == 0:
                    raise ValueError(f"Empty token array for key {key} in {part_path}:{line_number}")

                output_path = token_dir / f"{key}.npy"
                if key in seen_keys:
                    existing_array = np.load(output_path, allow_pickle=False)
                    if existing_array.ndim != 1:
                        raise ValueError(f"Existing token array is not 1D for duplicate key {key}")
                    if np.array_equal(existing_array, array):
                        duplicate_same_count += 1
                        continue
                    raise ValueError(f"Conflicting duplicate key across parts: {key}")

                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, array)
                seen_keys.add(key)
                written += 1

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    if expected_keys is not None:
        missing_keys = sorted(expected_keys - seen_keys)
        unexpected_keys = sorted(seen_keys - expected_keys)
        if missing_keys or unexpected_keys:
            problems = []
            if missing_keys:
                problems.append(f"missing={missing_keys[:10]}")
            if unexpected_keys:
                problems.append(f"unexpected={unexpected_keys[:10]}")
            raise ValueError(
                "Mismatch between parts output and expected scp keys: " + "; ".join(problems)
            )

    return {
        "parts_dir": parts_dir,
        "token_dir": token_dir,
        "written_count": written,
        "duplicate_same_count": duplicate_same_count,
        "expected_count": 0 if expected_keys is None else len(expected_keys),
        "missing_count": len(missing_keys),
        "unexpected_count": len(unexpected_keys),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert s3tokenizer distributed JSONL parts into per-sample .npy files."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to repository root.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Directory containing part_*_of_* JSONL files.",
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=None,
        help="Final token directory. Defaults to data/{dataset}/token/{model_output_dir}.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help="s3tokenizer model name used to choose default parts/token directories.",
    )
    parser.add_argument(
        "--expected-scp",
        type=Path,
        default=None,
        help="Optional wav.scp used to validate expected keys.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    parts_dir = args.parts_dir or get_default_parts_dir(project_root, args.dataset, args.model)
    token_dir = args.token_dir or get_default_token_dir(project_root, args.dataset, args.model)
    summary = materialize_parts(parts_dir=parts_dir, token_dir=token_dir, expected_scp=args.expected_scp)
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary.items()
    }
    print(json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
