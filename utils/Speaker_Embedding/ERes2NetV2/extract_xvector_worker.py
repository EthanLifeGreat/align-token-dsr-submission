"""Extract x-vector (worker version for parallel processing).

This script is designed for parallel processing across multiple GPUs.
Each worker processes a subset of data based on worker_id.

Usage:
    CUDA_VISIBLE_DEVICES=0 python extract_xvector_worker.py \
        --num-workers 8 \
        --worker-id 0 \
        --dataset AISHELL-2

Features:
    - Progress tracking with tqdm
    - Log file per worker in ./logs/
    - Resume from checkpoint (skip existing files)
"""

import os
import sys
import csv
import pathlib
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

speakerlab_path = os.environ.get("SPEAKERLAB_PATH")
if speakerlab_path:
    sys.path.insert(0, speakerlab_path)

# Add project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from speakerlab.process.processor import FBank
except ImportError as e:
    print(f"Warning: Cannot import speakerlab: {e}")
    FBank = None

from modelscope.hub.snapshot_download import snapshot_download


class SpeakerVerification:
    """Speaker verification class for x-vector extraction."""

    SUPPORTS = {
        'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
            'revision': 'v1.0.1',
            'model_config': {
                'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
                'args': {
                    'feat_dim': 80,
                    'embedding_size': 192,
                    'baseWidth': 24,
                    'scale': 4,
                    'expansion': 4,
                }
            },
            'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
        },
    }

    def __init__(self, model_id: str, local_model_dir: str = 'pretrained',
                 device: str = 'cuda', logger: logging.Logger = None):
        self.model_id = model_id
        self.local_model_dir = pathlib.Path(local_model_dir)
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)

        self.conf = self.SUPPORTS[model_id]
        self.model = None
        self.feature_extractor = None
        self._load_model()

    def _load_model(self):
        save_dir = self.local_model_dir / self.model_id.split('/')[1]
        save_dir.mkdir(exist_ok=True, parents=True)

        model_path = save_dir / self.conf['model_pt']
        if not model_path.exists():
            self.logger.info(f"Downloading model {self.model_id}...")
            cache_dir = snapshot_download(self.model_id, revision=self.conf['revision'])
            cache_dir = pathlib.Path(cache_dir)

            for src in cache_dir.glob('*'):
                if self.conf['model_pt'] in src.name:
                    dst = save_dir / src.name
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.absolute())
                    self.logger.info(f"Linked model file: {dst} -> {src}")

        pretrained_state = torch.load(model_path, map_location='cpu', weights_only=False)
        model_class = self._dynamic_import(self.conf['model_config']['obj'])
        self.model = model_class(**self.conf['model_config']['args'])

        model_state = self.model.state_dict()
        loaded_state = {}

        for k, v in pretrained_state.items():
            if k in model_state:
                loaded_state[k] = v
            else:
                new_k = k.replace('model.', '')
                if new_k in model_state:
                    loaded_state[new_k] = v
                elif new_k.replace('backbone.', '') in model_state:
                    loaded_state[new_k.replace('backbone.', '')] = v

        if loaded_state:
            self.model.load_state_dict(loaded_state, strict=False)
        else:
            self.model.load_state_dict(pretrained_state, strict=False)

        self.model.to(self.device)
        self.model.eval()

        if FBank is not None:
            self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        else:
            raise ImportError("Cannot initialize FBank feature extractor")

        self.logger.info(f"Model loaded on device: {self.device}")

    def _dynamic_import(self, import_path: str):
        module_path, class_name = import_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def extract_embedding(self, wav_file: str, normalize: bool = True) -> np.ndarray:
        """Extract x-vector embedding from audio file."""
        import torchaudio
        import warnings

        # Load audio
        try:
            wav, fs = torchaudio.load(wav_file)
            if wav.numel() == 0 or wav.shape[1] == 0:
                raise ValueError(f"Audio file {wav_file} is empty")
            if fs != 16000:
                wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=16000)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {wav_file}: {e}")

        if wav.shape[1] < 1600:
            warnings.warn(f"Audio {wav_file} too short ({wav.shape[1] / 16000:.2f}s)")

        # Extract features
        try:
            feat = self.feature_extractor(wav)
            if feat.shape[0] < 10:
                raise ValueError(f"Audio {wav_file} has insufficient frames ({feat.shape[0]})")
            feat = feat.unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed {wav_file}: {e}")

        # Model inference
        with torch.no_grad():
            try:
                embedding = self.model(feat)
                if embedding.dim() > 2:
                    embedding = embedding.squeeze(0)
                embedding = embedding.detach().cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Model inference failed {wav_file}: {e}")

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding


def setup_logger(worker_id: int, log_dir: pathlib.Path) -> logging.Logger:
    """Setup logger for worker."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"xvector_worker_{worker_id}_{timestamp}.log"

    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        f"[Worker {worker_id}] %(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(f"[Worker {worker_id}] %(message)s"))
    logger.addHandler(ch)

    return logger


def count_meta_rows(meta_csv: pathlib.Path) -> int:
    """Quickly count rows in meta.csv."""
    with open(meta_csv, 'r') as f:
        return sum(1 for _ in f) - 1  # Subtract header


def iterate_meta(meta_csv: pathlib.Path, wav_dir: pathlib.Path, 
                 num_workers: int, worker_id: int):
    """Iterate over meta.csv rows assigned to this worker."""
    with open(meta_csv, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx % num_workers == worker_id:
                spk_id = row['spk_id']
                wav_id = row['wav_id']
                wav_path = wav_dir / spk_id / f'{wav_id}.wav'
                if wav_path.exists():
                    yield wav_path, spk_id, wav_id


def main():
    parser = argparse.ArgumentParser(description="Extract x-vector (Worker)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (AISHELL-2, LibriTTS, etc.)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (GPU is set via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--model-id", type=str,
                        default="iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
                        help="Model ID for speaker verification")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root path")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory (overrides $XVECTOR_LOG_DIR and default project_root/logs/xvector)")
    # Worker-specific arguments
    parser.add_argument("--num-workers", type=int, required=True,
                        help="Total number of workers")
    parser.add_argument("--worker-id", type=int, required=True,
                        help="Worker ID (0 to num_workers-1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")

    args = parser.parse_args()

    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id must be in [0, {args.num_workers-1}], got {args.worker_id}")

    # Get project root
    if args.project_root:
        project_root = pathlib.Path(args.project_root)
    else:
        project_root = PROJECT_ROOT

    # Setup paths
    data_dir = project_root / "data" / args.dataset
    meta_csv = data_dir / "meta.csv"
    wav_dir = data_dir / "wav"
    xvec_dir = data_dir / "xvector" / "eresnet2v2"

    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_csv}")

    xvec_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    if args.log_dir:
        log_dir = pathlib.Path(args.log_dir)
    elif os.environ.get("XVECTOR_LOG_DIR"):
        log_dir = pathlib.Path(os.environ["XVECTOR_LOG_DIR"])
    else:
        log_dir = project_root / "logs" / "xvector"
    logger = setup_logger(args.worker_id, log_dir)

    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    logger.info(f"Starting x-vector extraction")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output directory: {xvec_dir}")
    logger.info(f"  GPU: CUDA_VISIBLE_DEVICES={gpu_id}")
    logger.info(f"  Worker: {args.worker_id}/{args.num_workers}")

    # Load model
    logger.info("Loading model...")
    model = SpeakerVerification(
        model_id=args.model_id,
        device=args.device,
        logger=logger
    )

    # Quick count of total rows for progress estimation
    total_rows = count_meta_rows(meta_csv)
    estimated_total = total_rows // args.num_workers
    logger.info(f"Total rows in meta.csv: {total_rows}, estimated for this worker: {estimated_total}")

    # Extract x-vectors
    import time
    from collections import deque

    start_time = time.time()
    last_log_time = start_time
    last_log_count = 0
    success_count = 0
    skip_count = 0
    fail_count = 0
    processed_count = 0

    # Sliding window for ETA calculation (like tqdm)
    speed_window = deque(maxlen=10)  # Store last 10 speeds
    log_interval = max(1, estimated_total // 1000)  # Log every .1%

    for wav_path, spk_id, wav_id in tqdm(
        iterate_meta(meta_csv, wav_dir, args.num_workers, args.worker_id),
        desc=f"Worker {args.worker_id}",
        total=estimated_total
    ):
        processed_count += 1
        spk_xvec_dir = xvec_dir / spk_id
        spk_xvec_dir.mkdir(parents=True, exist_ok=True)
        xvec_path = spk_xvec_dir / f'{wav_id}.npy'

        # Skip existing files only if they can be loaded successfully (resume support)
        if xvec_path.exists() and not args.overwrite:
            try:
                # Verify the existing file is valid by attempting to load it
                np.load(xvec_path, allow_pickle=False)
                skip_count += 1
                continue
            except (ValueError, EOFError, OSError) as e:
                # File exists but is corrupted, will be overwritten
                logger.warning(f"Corrupted x-vector file {xvec_path}: {e}, will re-extract")
        try:
            embedding = model.extract_embedding(str(wav_path), normalize=True)
            xvec_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(xvec_path, embedding)
            success_count += 1
        except Exception as e:
            fail_count += 1
            logger.error(f"Failed to process {wav_path}: {e}")

        # Log progress periodically with ETA
        if processed_count % log_interval == 0:
            current_time = time.time()
            # Calculate speed since last log
            time_since_last_log = current_time - last_log_time
            count_since_last_log = processed_count - last_log_count
            instant_speed = count_since_last_log / time_since_last_log if time_since_last_log > 0 else 0

            # Update sliding window
            speed_window.append(instant_speed)
            avg_speed = sum(speed_window) / len(speed_window) if speed_window else 0

            # Calculate ETA using sliding window speed
            remaining = estimated_total - processed_count
            eta_seconds = remaining / avg_speed if avg_speed > 0 else 0
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            eta_s = int(eta_seconds % 60)

            progress = processed_count / estimated_total * 100
            elapsed = current_time - start_time
            elapsed_h = int(elapsed // 3600)
            elapsed_m = int((elapsed % 3600) // 60)
            elapsed_s = int(elapsed % 60)

            logger.info(f"Progress: {processed_count}/{estimated_total} ({progress:.1f}%) | "
                       f"Elapsed: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d} | "
                       f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d} | "
                       f"Speed: {avg_speed:.1f} files/s | "
                       f"Success: {success_count}, Skip: {skip_count}, Fail: {fail_count}")

            last_log_time = current_time
            last_log_count = processed_count

    elapsed = time.time() - start_time
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)
    elapsed_s = int(elapsed % 60)
    logger.info(f"Completed! Total time: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d} | "
               f"Processed: {processed_count}, Success: {success_count}, Skip: {skip_count}, Fail: {fail_count}")


if __name__ == "__main__":
    main()
