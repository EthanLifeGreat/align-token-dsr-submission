from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from src.external_baselines.common import ManifestItem, normalize_asr_text, validate_disjoint_splits, wav_duration


ROOT = Path(__file__).resolve().parents[1]


def write_wave(path: Path, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ExternalBaselineTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args], cwd=ROOT, text=True, capture_output=True, check=True
        )

    def test_normalization_and_split_leakage(self) -> None:
        self.assertEqual(normalize_asr_text("你 好，A-1!"), "你好A1")
        source = Path("placeholder.wav")
        with self.assertRaises(ValueError):
            validate_disjoint_splits(
                {
                    "train": [ManifestItem("train", "speaker_a", source, "a")],
                    "validation": [ManifestItem("validation", "speaker_a", source, "b")],
                    "test": [ManifestItem("test", "speaker_b", source, "c")],
                }
            )

    def test_prepare_cer_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_wav, validation_wav, test_wav = (root / name for name in ("train.wav", "validation.wav", "test.wav"))
            for path in (train_wav, validation_wav, test_wav):
                write_wave(path)
            manifests = {}
            for split, wav, speaker in [
                ("train", train_wav, "speaker_train"),
                ("validation", validation_wav, "speaker_validation"),
                ("test", test_wav, "speaker_test"),
            ]:
                manifest = root / f"{split}.csv"
                write_csv(
                    manifest,
                    [{"utt_id": f"dummy_{split}", "speaker_id": speaker, "source_wav": str(wav), "text": "你好 A1"}],
                    ["utt_id", "speaker_id", "source_wav", "text"],
                )
                manifests[split] = manifest
            prepared = root / "prepared"
            self.run_script(
                "scripts/prepare_asr_tts_manifests.py",
                "--train-manifest", str(manifests["train"]),
                "--validation-manifest", str(manifests["validation"]),
                "--test-manifest", str(manifests["test"]),
                "--output-dir", str(prepared),
            )
            prepared_test = read_rows(prepared / "test.csv")[0]
            self.assertEqual(prepared_test["normalized_text"], "你好A1")

            ref = root / "ref.txt"
            hyp = root / "hyp.txt"
            ref.write_text("dummy_test 你好\n", encoding="utf-8")
            hyp.write_text("dummy_test 你坏\n", encoding="utf-8")
            cer_summary = root / "cer.csv"
            cer_per_utt = root / "cer_per_utt.csv"
            self.run_script(
                "scripts/compute_cer.py", "--ref", str(ref), "--hyp", str(hyp),
                "--output", str(cer_summary), "--per-utt-output", str(cer_per_utt),
            )
            cer_row = read_rows(cer_per_utt)[0]
            self.assertEqual(cer_row["S"], "1")

            generated = root / "generated.wav"
            write_wave(generated)
            baseline_manifest = root / "manifest.csv"
            write_csv(
                baseline_manifest,
                [{"utt_id": "dummy_test", "speaker_id": "speaker_test", "status": "success", "generated_wav": str(generated)}],
                ["utt_id", "speaker_id", "status", "generated_wav"],
            )
            quality = root / "quality.csv"
            similarity = root / "similarity.csv"
            write_csv(quality, [{"wav_id": "dummy_test", "utmos": 3.0, "dnsmos_ovrl": 2.5}], ["wav_id", "utmos", "dnsmos_ovrl"])
            write_csv(similarity, [{"utt_id": "dummy_test", "source_sim": 0.1, "cond_sim": 0.8}], ["utt_id", "source_sim", "cond_sim"])
            metrics_dir = root / "metrics"
            self.run_script(
                "scripts/aggregate_baseline_metrics.py", "--manifest", str(baseline_manifest), "--method", "dummy",
                "--cer-per-utt-csv", str(cer_per_utt), "--quality-csv", str(quality), "--speaker-csv", str(similarity),
                "--output-dir", str(metrics_dir),
            )
            summary = json.loads((metrics_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_empty_count"], 0)
            self.assertAlmostEqual(summary["cer_pct"], 50.0)

    def test_seed_worker_protocol_and_failure_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_ok, source_fail, target = root / "source_ok.wav", root / "source_fail.wav", root / "target.wav"
            for path in (source_ok, source_fail, target):
                write_wave(path)
            manifest = root / "test.csv"
            write_csv(
                manifest,
                [
                    {"utt_id": "ok", "speaker_id": "speaker_a", "source_wav": str(source_ok), "text": "dummy text"},
                    {"utt_id": "fail", "speaker_id": "speaker_b", "source_wav": str(source_fail), "text": "dummy text"},
                ],
                ["utt_id", "speaker_id", "source_wav", "text"],
            )
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys, wave\n"
                "from pathlib import Path\n"
                "for line in sys.stdin:\n"
                "    request = json.loads(line)\n"
                "    assert 'reference_text' not in request and 'text' not in request\n"
                "    if request['utt_id'] == 'fail':\n"
                "        response = {'utt_id': 'fail', 'status': 'failed', 'error': 'expected test failure'}\n"
                "    else:\n"
                "        output = Path(request['output_wav']); output.parent.mkdir(parents=True, exist_ok=True)\n"
                "        with wave.open(str(output), 'wb') as handle:\n"
                "            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(b'\\x00\\x00' * 1600)\n"
                "        response = {'utt_id': request['utt_id'], 'status': 'success', 'error': '', 'seed_vc_model': request['seed_vc_model'], 'seed_vc_source_revision': request['seed_vc_source_revision'], 'seed_vc_model_revision': request['seed_vc_model_revision']}\n"
                "    print(json.dumps(response), flush=True)\n",
                encoding="utf-8",
            )
            output_dir = root / "seed_output"
            self.run_script(
                "scripts/run_seed_vc_normal_ref.py", "--test-manifest", str(manifest), "--target-wav", str(target),
                "--seed-vc-worker", f"{sys.executable} {worker}", "--output-dir", str(output_dir), "--worker-timeout-sec", "10",
            )
            rows = {row["utt_id"]: row for row in read_rows(output_dir / "manifest.csv")}
            self.assertEqual(rows["ok"]["status"], "success")
            self.assertEqual(rows["fail"]["status"], "vc_failed")
            self.assertAlmostEqual(wav_duration(Path(rows["fail"]["generated_wav"])), 0.5, places=3)
            config = (output_dir / "run_config.json").read_text(encoding="utf-8")
            self.assertNotIn(str(worker), config)


if __name__ == "__main__":
    unittest.main()
