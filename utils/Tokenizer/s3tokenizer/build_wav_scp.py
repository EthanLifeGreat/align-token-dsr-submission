from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    count_lines,
    get_default_scp_path,
    iter_meta_rows,
    resolve_project_root,
)


def build_wav_scp(
    project_root: Path,
    dataset: str,
) -> dict[str, object]:
    default_scp_path = get_default_scp_path(project_root, dataset)
    meta_row_count = sum(1 for _ in iter_meta_rows(project_root, dataset))
    existing_scp_rows = count_lines(default_scp_path) if default_scp_path.exists() else 0
    trusted_existing = default_scp_path.exists() and existing_scp_rows == meta_row_count
    kaldi_built = False

    if not trusted_existing:
        build_kaldi_sh = (
            project_root / "utils" / "Data_Stats" / "build_kaldi" / "build_kaldi.sh"
        )
        if not build_kaldi_sh.exists():
            raise FileNotFoundError(f"build_kaldi.sh not found: {build_kaldi_sh}")
        subprocess.run(
            ["bash", str(build_kaldi_sh), dataset],
            cwd=str(project_root),
            check=True,
        )
        if not default_scp_path.exists():
            raise FileNotFoundError(
                f"kaldi wav.scp still not found after build_kaldi for {dataset}: {default_scp_path}"
            )
        kaldi_built = True
    scp_row_count = count_lines(default_scp_path)
    if scp_row_count != meta_row_count:
        raise ValueError(
            f"wav.scp row count mismatch for {dataset}: "
            f"meta_rows={meta_row_count}, wav_scp_rows={scp_row_count}"
        )
    return {
        "dataset": dataset,
        "kaldi_built": kaldi_built,
        "trusted_existing": trusted_existing,
        "scp_path": default_scp_path,
        "total_meta_rows": meta_row_count,
        "total_scp_rows": scp_row_count,
        "planned_count": scp_row_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or rebuild kaldi/wav.scp for s3tokenizer extraction."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    summary = build_wav_scp(
        project_root=project_root,
        dataset=args.dataset,
    )
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary.items()
    }
    print(json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
