from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATASETS = (
    "AISHELL-2",
    "MAGICDATA-cut",
    "CSMSC",
    "LibriTTS-16k",
    "THCHS30",
)
SUPPORTED_MODELS = (
    "speech_tokenizer_v1",
    "speech_tokenizer_v1_25hz",
    "speech_tokenizer_v2_25hz",
    "speech_tokenizer_v3_25hz",
)
DEFAULT_MODEL = "speech_tokenizer_v1"
MODEL_OUTPUT_DIRS = {
    "speech_tokenizer_v1": "s3tokenizer_v1",
    "speech_tokenizer_v1_25hz": "s3tokenizer_v1_25hz",
    "speech_tokenizer_v2_25hz": "s3tokenizer_v2_25hz",
    "speech_tokenizer_v3_25hz": "s3tokenizer_v3_25hz",
}


@dataclass(frozen=True)
class SamplePaths:
    key: str
    wav_path: Path
    token_path: Path


def resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is None:
        return PROJECT_ROOT
    return Path(project_root).resolve()


def get_dataset_dir(project_root: Path, dataset: str) -> Path:
    return project_root / "data" / dataset


def get_default_scp_path(project_root: Path, dataset: str) -> Path:
    return get_dataset_dir(project_root, dataset) / "kaldi" / "wav.scp"


def get_model_output_dir(model: str) -> str:
    try:
        return MODEL_OUTPUT_DIRS[model]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model: {model}. Expected one of: {', '.join(SUPPORTED_MODELS)}"
        ) from exc


def get_default_token_dir(project_root: Path, dataset: str, model: str = DEFAULT_MODEL) -> Path:
    return get_dataset_dir(project_root, dataset) / "token" / get_model_output_dir(model)


def get_default_parts_dir(project_root: Path, dataset: str, model: str = DEFAULT_MODEL) -> Path:
    return project_root / "exp" / get_model_output_dir(model) / dataset / "parts"


def get_default_log_path(project_root: Path, dataset: str, model: str = DEFAULT_MODEL) -> Path:
    return project_root / "exp" / get_model_output_dir(model) / dataset / "run.log"


def get_default_summary_path(
    project_root: Path,
    dataset: str,
    model: str = DEFAULT_MODEL,
) -> Path:
    return project_root / "exp" / get_model_output_dir(model) / dataset / "summary.json"


def iter_meta_rows(project_root: Path, dataset: str) -> Iterator[dict[str, str]]:
    meta_path = get_dataset_dir(project_root, dataset) / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def build_sample_paths(
    project_root: Path,
    dataset: str,
    row: dict[str, str],
    token_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
) -> SamplePaths:
    dataset_dir = get_dataset_dir(project_root, dataset)
    token_root = token_dir or get_default_token_dir(project_root, dataset, model)
    wav_rel = (Path(row["spk_id"]) / row["wav_id"]).with_suffix(".wav")
    key = wav_rel.with_suffix("").as_posix()
    wav_path = dataset_dir / "wav" / wav_rel
    token_path = token_root / f"{key}.npy"
    return SamplePaths(key=key, wav_path=wav_path, token_path=token_path)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_scp_entries(path: Path) -> Iterator[tuple[str, Path]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                key, wav_path = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Invalid wav.scp line {line_number}: {raw_line!r}") from exc
            yield key, Path(wav_path)


def collect_scp_keys(path: Path) -> set[str]:
    return {key for key, _ in iter_scp_entries(path)}


def iter_token_files(token_dir: Path) -> Iterator[tuple[str, Path]]:
    if not token_dir.exists():
        return
    for npy_path in sorted(token_dir.rglob("*.npy")):
        key = npy_path.relative_to(token_dir).with_suffix("").as_posix()
        yield key, npy_path


def collect_token_keys(token_dir: Path) -> set[str]:
    return {key for key, _ in iter_token_files(token_dir)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def serialize_paths(payload: dict) -> dict:
    serialized = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, dict):
            serialized[key] = serialize_paths(value)
        elif isinstance(value, list):
            serialized[key] = [
                str(item) if isinstance(item, Path) else asdict(item) if hasattr(item, "__dataclass_fields__") else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def preview_items(items: Iterable[str], limit: int = 5) -> list[str]:
    preview = []
    for item in items:
        preview.append(item)
        if len(preview) >= limit:
            break
    return preview
