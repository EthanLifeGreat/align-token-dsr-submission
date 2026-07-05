from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_wav_scp import build_wav_scp  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_DATASETS,
    SUPPORTED_MODELS,
    build_sample_paths,
    collect_scp_keys,
    collect_token_keys,
    count_lines,
    get_default_log_path,
    get_default_parts_dir,
    get_default_scp_path,
    get_default_summary_path,
    get_default_token_dir,
    get_model_output_dir,
    iter_meta_rows,
    preview_items,
    resolve_project_root,
    serialize_paths,
    write_json,
)
from materialize_parts import iter_part_paths, materialize_parts  # noqa: E402


PIPELINE_STEPS = ("generate-scp", "infer", "materialize", "validate")


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_log(log_path: Path, dataset: str, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{iso_now()}] [{dataset}] {message}\n")


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_summary(summary_path: Path, summary: dict) -> None:
    write_json(summary_path, serialize_paths(summary))


def get_dataset_exp_dir(project_root: Path, dataset: str, model: str) -> Path:
    return project_root / "exp" / get_model_output_dir(model) / dataset


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def find_latest_parts_dir(parts_root: Path) -> Path | None:
    if not parts_root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    direct_parts = list(parts_root.glob("part_*_of_*"))
    if direct_parts:
        latest_mtime = max(path.stat().st_mtime for path in direct_parts)
        candidates.append((latest_mtime, parts_root))
    for child in parts_root.iterdir():
        if not child.is_dir():
            continue
        part_paths = list(child.glob("part_*_of_*"))
        if not part_paths:
            continue
        latest_mtime = max(path.stat().st_mtime for path in part_paths)
        candidates.append((latest_mtime, child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_expected_meta_keys(project_root: Path, dataset: str) -> set[str]:
    keys: set[str] = set()
    for row in iter_meta_rows(project_root, dataset):
        sample = build_sample_paths(project_root, dataset, row)
        if sample.key in keys:
            raise ValueError(f"Duplicate key in meta.csv for {dataset}: {sample.key}")
        keys.add(sample.key)
    return keys


def validate_tokens(
    project_root: Path,
    dataset: str,
    token_dir: Path,
    scp_path: Path,
    full_dataset: bool,
) -> dict[str, object]:
    if full_dataset:
        expected_keys = load_expected_meta_keys(project_root, dataset)
    else:
        expected_keys = collect_scp_keys(scp_path)

    token_keys = collect_token_keys(token_dir)
    missing_keys = sorted(expected_keys - token_keys)
    unexpected_keys = sorted(token_keys - expected_keys) if full_dataset else []

    if missing_keys or unexpected_keys:
        problems = []
        if missing_keys:
            problems.append(f"missing={preview_items(missing_keys)}")
        if unexpected_keys:
            problems.append(f"unexpected={preview_items(unexpected_keys)}")
        raise ValueError(
            f"Token key validation failed for {dataset}: " + "; ".join(problems)
        )

    checked = 0
    empty_keys: list[str] = []
    bad_shape_keys: list[str] = []
    bad_dtype_keys: list[str] = []
    for key in sorted(expected_keys):
        npy_path = token_dir / f"{key}.npy"
        array = np.load(npy_path, allow_pickle=False)
        checked += 1
        if array.ndim != 1:
            bad_shape_keys.append(key)
        if array.size == 0:
            empty_keys.append(key)
        if array.dtype != np.int32:
            bad_dtype_keys.append(key)

    if bad_shape_keys or empty_keys or bad_dtype_keys:
        problems = []
        if bad_shape_keys:
            problems.append(f"non_1d={preview_items(bad_shape_keys)}")
        if empty_keys:
            problems.append(f"empty={preview_items(empty_keys)}")
        if bad_dtype_keys:
            problems.append(f"non_int32={preview_items(bad_dtype_keys)}")
        raise ValueError(
            f"Token payload validation failed for {dataset}: " + "; ".join(problems)
        )

    return {
        "dataset": dataset,
        "token_dir": token_dir,
        "validated_count": checked,
        "full_dataset": full_dataset,
        "expected_count": len(expected_keys),
        "token_file_count": len(token_keys),
        "scp_count": count_lines(scp_path),
    }


def build_infer_command(
    scp_path: Path,
    parts_dir: Path,
    visible_gpus: str,
    nproc_per_node: int,
    batch_size: int,
    model: str,
    torchrun_path: Path,
    s3tokenizer_path: Path,
) -> str:
    if nproc_per_node <= 0:
        raise ValueError("nproc_per_node must be positive.")
    rdzv_id = f"s3tokenizer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return " ".join(
        [
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible_gpus)};",
            shlex.quote(str(torchrun_path)),
            f"--nproc_per_node={nproc_per_node}",
            "--nnodes=1",
            f"--rdzv_id={shlex.quote(rdzv_id)}",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            shlex.quote(str(s3tokenizer_path)),
            "--wav_scp",
            shlex.quote(str(scp_path)),
            "--device",
            "cuda",
            "--output_dir",
            shlex.quote(str(parts_dir)),
            "--batch_size",
            str(batch_size),
            "--model",
            shlex.quote(model),
        ]
    )


def resolve_env_executables(conda_env: str) -> tuple[Path, Path]:
    torchrun_path = shutil.which("torchrun")
    s3tokenizer_path = shutil.which("s3tokenizer")
    if torchrun_path and s3tokenizer_path:
        return Path(torchrun_path), Path(s3tokenizer_path)

    conda_path = shutil.which("conda")
    if not conda_path:
        raise FileNotFoundError(
            "Failed to resolve torchrun/s3tokenizer from current PATH, and `conda` is not available."
        )

    probe = subprocess.run(
        [
            conda_path,
            "run",
            "-n",
            conda_env,
            "python",
            "-c",
            (
                "import json, shutil; "
                "print(json.dumps({'torchrun': shutil.which('torchrun'), "
                "'s3tokenizer': shutil.which('s3tokenizer')}))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        stderr = probe.stderr.strip()
        stdout = probe.stdout.strip()
        raise RuntimeError(
            "Failed to resolve torchrun/s3tokenizer via "
            f"`conda run -n {conda_env}`. stdout={stdout!r} stderr={stderr!r}"
        )

    payload = json.loads(probe.stdout.strip().splitlines()[-1])
    torchrun_path = payload.get("torchrun")
    s3tokenizer_path = payload.get("s3tokenizer")
    if not torchrun_path or not s3tokenizer_path:
        raise FileNotFoundError(
            f"Failed to resolve torchrun/s3tokenizer inside conda env {conda_env}"
        )
    return Path(torchrun_path), Path(s3tokenizer_path)


def run_inference(
    dataset: str,
    project_root: Path,
    scp_path: Path,
    parts_dir: Path,
    gpus: str,
    batch_size: int,
    model: str,
    torchrun_path: Path,
    s3tokenizer_path: Path,
    log_path: Path,
) -> dict[str, object]:
    planned_count = count_lines(scp_path)
    if planned_count == 0:
        return {
            "dataset": dataset,
            "command": None,
            "planned_count": 0,
            "parts_dir": parts_dir,
            "started_at": iso_now(),
            "finished_at": iso_now(),
            "duration_seconds": 0.0,
            "returncode": 0,
            "part_files": 0,
            "skipped": True,
        }

    requested_gpu_ids = [item.strip() for item in gpus.split(",") if item.strip()]
    if not requested_gpu_ids:
        raise ValueError("At least one GPU id must be provided.")
    nproc_per_node = min(len(requested_gpu_ids), planned_count)
    visible_gpus = ",".join(requested_gpu_ids[:nproc_per_node])

    if parts_dir.exists() and any(parts_dir.iterdir()):
        raise FileExistsError(
            f"Parts directory already exists and is not empty: {parts_dir}. "
            "Use a new run id or remove the stale directory."
        )
    parts_dir.mkdir(parents=True, exist_ok=True)

    command = build_infer_command(
        scp_path=scp_path,
        parts_dir=parts_dir,
        visible_gpus=visible_gpus,
        nproc_per_node=nproc_per_node,
        batch_size=batch_size,
        model=model,
        torchrun_path=torchrun_path,
        s3tokenizer_path=s3tokenizer_path,
    )
    started_at = iso_now()
    append_log(log_path, dataset, f"infer command: {command}")
    with log_path.open("a", encoding="utf-8") as infer_log_handle:
        infer_log_handle.write(f"[{iso_now()}] [{dataset}] infer subprocess output begin\n")
        infer_log_handle.flush()
        process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=str(project_root),
            stdout=infer_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        returncode = process.wait()
        infer_log_handle.write(
            f"[{iso_now()}] [{dataset}] infer subprocess output end returncode={returncode}\n"
        )
        infer_log_handle.flush()
    finished_at = iso_now()
    if returncode != 0:
        append_log(log_path, dataset, f"infer failed returncode={returncode}")
        raise RuntimeError(
            f"s3tokenizer inference failed for {dataset} with exit code {returncode}"
        )
    part_files = len(iter_part_paths(parts_dir))
    return {
        "dataset": dataset,
        "command": command,
        "planned_count": planned_count,
        "parts_dir": parts_dir,
        "requested_gpus": gpus,
        "visible_gpus": visible_gpus,
        "nproc_per_node": nproc_per_node,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": (
            datetime.fromisoformat(finished_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp()
        ),
        "returncode": returncode,
        "part_files": part_files,
        "skipped": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the resumable multi-GPU S3Tokenizer extraction pipeline."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to process serially.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=PIPELINE_STEPS,
        default=list(PIPELINE_STEPS),
        help="Pipeline steps to run.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to repository root.",
    )
    parser.add_argument(
        "--gpus",
        default="1,3,4,5",
        help="Comma-separated GPU ids passed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-process batch size passed to s3tokenizer.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help="s3tokenizer model name.",
    )
    parser.add_argument(
        "--conda-env",
        default="s3tokenizer",
        help="Conda environment containing the s3tokenizer CLI.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id used for exp/{model_output_dir}/{dataset}/parts/{run_id}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    torchrun_path, s3tokenizer_path = resolve_env_executables(args.conda_env)

    for dataset in args.datasets:
        dataset_exp_dir = get_dataset_exp_dir(project_root, dataset, args.model)
        dataset_exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = get_default_log_path(project_root, dataset, args.model)
        summary_path = get_default_summary_path(project_root, dataset, args.model)
        scp_path = get_default_scp_path(project_root, dataset)
        token_dir = get_default_token_dir(project_root, dataset, args.model)
        parts_root = get_default_parts_dir(project_root, dataset, args.model)
        summary = load_summary(summary_path)
        summary.update(
            {
                "dataset": dataset,
                "model": args.model,
                "model_output_dir": get_model_output_dir(args.model),
                "project_root": project_root,
                "scp_path": scp_path,
                "token_dir": token_dir,
                "parts_root": parts_root,
                "updated_at": iso_now(),
            }
        )
        summary.setdefault("steps", {})
        append_log(log_path, dataset, f"starting model={args.model} steps={args.steps}")

        current_parts_dir: Path | None = None

        if "generate-scp" in args.steps:
            append_log(log_path, dataset, "running generate-scp")
            scp_summary = build_wav_scp(
                project_root=project_root,
                dataset=dataset,
            )
            summary["steps"]["generate-scp"] = {
                **serialize_paths(scp_summary),
                "started_at": iso_now(),
                "finished_at": iso_now(),
            }
            append_log(
                log_path,
                dataset,
                "generate-scp planned_count="
                f"{scp_summary['planned_count']} "
                f"trusted_existing={scp_summary['trusted_existing']} "
                f"kaldi_built={scp_summary['kaldi_built']}",
            )
            save_summary(summary_path, summary)

        if "infer" in args.steps:
            if not scp_path.exists():
                raise FileNotFoundError(f"wav.scp not found for {dataset}: {scp_path}")
            current_parts_dir = parts_root / (args.run_id or new_run_id())
            append_log(log_path, dataset, f"running infer parts_dir={current_parts_dir}")
            infer_summary = run_inference(
                dataset=dataset,
                project_root=project_root,
                scp_path=scp_path,
                parts_dir=current_parts_dir,
                gpus=args.gpus,
                batch_size=args.batch_size,
                model=args.model,
                torchrun_path=torchrun_path,
                s3tokenizer_path=s3tokenizer_path,
                log_path=log_path,
            )
            if not infer_summary["skipped"]:
                summary["parts_dir"] = current_parts_dir
            summary["steps"]["infer"] = serialize_paths(infer_summary)
            append_log(
                log_path,
                dataset,
                f"infer planned_count={infer_summary['planned_count']} part_files={infer_summary['part_files']}",
            )
            save_summary(summary_path, summary)

        if "materialize" in args.steps:
            if not scp_path.exists():
                raise FileNotFoundError(f"wav.scp not found for {dataset}: {scp_path}")
            scp_count = count_lines(scp_path)
            if scp_count == 0:
                materialize_summary = {
                    "parts_dir": None,
                    "token_dir": token_dir,
                    "written_count": 0,
                    "expected_count": 0,
                    "missing_count": 0,
                    "unexpected_count": 0,
                    "skipped": True,
                }
                summary["steps"]["materialize"] = serialize_paths(materialize_summary)
                append_log(log_path, dataset, "materialize skipped because wav.scp is empty")
                save_summary(summary_path, summary)
            else:
                if current_parts_dir is None:
                    summary_parts_dir = Path(summary["parts_dir"]) if summary.get("parts_dir") else None
                    if summary_parts_dir is not None and summary_parts_dir.exists():
                        current_parts_dir = summary_parts_dir
                    else:
                        current_parts_dir = find_latest_parts_dir(parts_root)
                if current_parts_dir is None:
                    raise FileNotFoundError(f"No parts directory found for {dataset}")
                append_log(log_path, dataset, f"running materialize parts_dir={current_parts_dir}")
                materialize_summary = materialize_parts(
                    parts_dir=current_parts_dir,
                    token_dir=token_dir,
                    expected_scp=scp_path,
                )
                summary["parts_dir"] = current_parts_dir
                summary["steps"]["materialize"] = serialize_paths(materialize_summary)
                append_log(
                    log_path,
                    dataset,
                    f"materialize written_count={materialize_summary['written_count']} "
                    f"missing_count={materialize_summary['missing_count']}",
                )
                save_summary(summary_path, summary)

        if "validate" in args.steps:
            full_dataset = True
            append_log(log_path, dataset, f"running validate full_dataset={full_dataset}")
            validate_summary = validate_tokens(
                project_root=project_root,
                dataset=dataset,
                token_dir=token_dir,
                scp_path=scp_path,
                full_dataset=full_dataset,
            )
            summary["steps"]["validate"] = serialize_paths(validate_summary)
            append_log(
                log_path,
                dataset,
                f"validate validated_count={validate_summary['validated_count']} "
                f"token_file_count={validate_summary['token_file_count']}",
            )
            save_summary(summary_path, summary)

        summary["updated_at"] = iso_now()
        save_summary(summary_path, summary)
        append_log(log_path, dataset, "finished successfully")


if __name__ == "__main__":
    main()
