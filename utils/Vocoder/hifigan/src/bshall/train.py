import argparse
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from hifigan.generator import HifiganGenerator
from hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from hifigan.dataset import MelDataset
from hifigan.mel_spec import LogMelSpectrogram
from hifigan.utils import load_checkpoint, save_checkpoint, plot_spectrogram


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BATCH_SIZE = 8
SEGMENT_LENGTH = 8320
HOP_LENGTH = 320

SAMPLE_RATE = 16000
BASE_LEARNING_RATE = 2e-4
FINETUNE_LEARNING_RATE = 1e-4
BETAS = (0.8, 0.99)
LEARNING_RATE_DECAY = 0.999
WEIGHT_DECAY = 1e-5
EPOCHS = 3100
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 5000
NUM_GENERATED_EXAMPLES = 10
CHECKPOINT_INTERVAL = 5000


def center_crop_waveform(waveform, target_length):
    extra = waveform.size(-1) - target_length
    if extra < 0:
        raise ValueError(
            f"Generated waveform is shorter than target: "
            f"{waveform.size(-1)} < {target_length}"
        )
    left = extra // 2
    return waveform[..., left : left + target_length]


def train_model(rank, world_size, args):
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://localhost:54322",
        timeout=timedelta(hours=1),
    )

    log_dir = Path(args.checkpoint_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    generator = HifiganGenerator().to(rank)
    discriminator = HifiganDiscriminator().to(rank)

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    optimizer_generator = optim.AdamW(
        generator.parameters(),
        lr=args.learning_rate,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_discriminator = optim.AdamW(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_generator = optim.lr_scheduler.ExponentialLR(
        optimizer_generator, gamma=LEARNING_RATE_DECAY
    )
    scheduler_discriminator = optim.lr_scheduler.ExponentialLR(
        optimizer_discriminator, gamma=LEARNING_RATE_DECAY
    )

    # Parse multiple dataset configurations
    # Each dataset should have: csv_path, wav_dir, mel_dir (optional)
    dataset_configs = []
    if args.datasets:
        for dataset_str in args.datasets:
            parts = dataset_str.split(':')
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(f"Invalid dataset format: {dataset_str}. Expected csv_path:wav_dir[:mel_dir]")

            csv_path = Path(parts[0])
            wav_dir = Path(parts[1])
            mel_dir = Path(parts[2]) if len(parts) > 2 else None

            dataset_configs.append({
                'csv_path': csv_path,
                'wav_data_dir': str(wav_dir),
                'mel_data_dir': str(mel_dir) if mel_dir else None,
            })
    else:
        # Fallback to old single dataset format for backward compatibility
        dataset_configs = [{
            'csv_path': args.csv_path,
            'wav_data_dir': str(args.wav_dataset_dir),
            'mel_data_dir': str(args.mel_dataset_dir) if args.mel_dataset_dir else None,
        }]

    # Load and merge all datasets, adding wav_data_dir and mel_data_dir columns
    # This avoids using ConcatDataset which has serialization issues with spawn mode
    all_train_dfs = []
    all_valid_dfs = []

    for cfg in dataset_configs:
        data_csv = pd.read_csv(cfg['csv_path'], dtype=str)
        # Add directory columns to each row
        data_csv['wav_data_dir'] = cfg['wav_data_dir']
        data_csv['mel_data_dir'] = cfg['mel_data_dir'] if cfg['mel_data_dir'] else ''

        # Keep the held-out test split out of vocoder training.
        train_mask = data_csv['split'].isin(['train'])
        train_df = data_csv[train_mask].reset_index(drop=True)
        if len(train_df) > 0:
            all_train_dfs.append(train_df)
            logger.info(f"Dataset {cfg['csv_path']}: {len(train_df)} train samples")

        # Extract validation data (dev or valid) for validation
        valid_mask = data_csv['split'].isin(['dev', 'valid'])
        valid_df = data_csv[valid_mask].reset_index(drop=True)
        if len(valid_df) > 0:
            all_valid_dfs.append(valid_df)
            logger.info(f"Dataset {cfg['csv_path']}: {len(valid_df)} dev+valid samples")

    # Concatenate all train and validation data
    if len(all_train_dfs) == 0:
        raise ValueError("No training data found in any dataset")

    train_df = pd.concat(all_train_dfs, ignore_index=True)

    if len(all_valid_dfs) > 0:
        valid_df = pd.concat(all_valid_dfs, ignore_index=True)
    else:
        raise ValueError("No validation data found in any dataset (need 'dev' or 'valid' split)")

    if args.validation_subset > 0 and len(valid_df) > args.validation_subset:
        valid_df = valid_df.sample(
            n=args.validation_subset,
            random_state=args.validation_seed,
        ).sort_values(["spk_id", "wav_id"]).reset_index(drop=True)

    logger.info(f"Total training samples: {len(train_df)}")
    logger.info(f"Validation samples before DDP sharding: {len(valid_df)}")

    # Use None for wav_data_dir/mel_data_dir since they're now in the DataFrame
    train_dataset = MelDataset(
        id2path_df=train_df,
        wav_data_dir=None,
        mel_data_dir=None,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=True,
        input_mel_source=args.input_mel_source,
        disable_augment=args.disable_aug,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    validation_dataset = MelDataset(
        id2path_df=valid_df,
        wav_data_dir=None,
        mel_data_dir=None,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=False,
        input_mel_source=args.input_mel_source,
        disable_augment=True,
    )
    validation_indices = list(range(rank, len(validation_dataset), world_size))
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    melspectrogram = LogMelSpectrogram().to(rank)

    if args.resume is not None:
        (
            global_step,
            best_loss,
            bad_validations,
            early_stopping_best_metric,
        ) = load_checkpoint(
            load_path=args.resume,
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            rank=rank,
            logger=logger,
            reset_optim=args.reset_optim,
        )
    else:
        global_step, best_loss = 0, float("inf")
        bad_validations, early_stopping_best_metric = 0, float("inf")

    if args.reset_optim:
        global_step, best_loss = 0, float("inf")
        bad_validations, early_stopping_best_metric = 0, float("inf")

    n_epochs = args.epochs
    total_train_batches = len(train_loader)
    start_epoch = global_step // max(total_train_batches, 1) + 1

    logger.info("**" * 40)
    logger.info(f"batch size per GPU: {args.batch_size}")
    logger.info(f"global batch size: {args.batch_size * world_size}")
    logger.info(f"iterations per epoch: {total_train_batches}")
    logger.info(f"total of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info(f"number of datasets: {len(dataset_configs)}")
    logger.info(f"input mel source: {args.input_mel_source}")
    logger.info(f"reset optimizer state: {args.reset_optim}")
    logger.info(f"disable waveform augmentation: {args.disable_aug}")
    logger.info(f"maximum steps: {args.max_steps}")
    logger.info(
        f"validation: every {args.validation_interval} steps, "
        f"subset={len(valid_df)}, patience={args.early_stopping_patience}, "
        f"min_delta={args.early_stopping_min_delta}, "
        f"min_steps={args.early_stopping_min_steps}"
    )
    for i, cfg in enumerate(dataset_configs):
        logger.info(f"  dataset {i+1}: csv={cfg['csv_path']}, wav_dir={cfg['wav_data_dir']}, mel_dir={cfg['mel_data_dir']}")
    logger.info("**" * 40 + "\n")

    stop = False
    last_validation_loss = best_loss
    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()
        average_loss_mel = average_loss_discriminator = average_loss_generator = 0
        for i, (wavs, mels, tgts) in enumerate(train_loader, 1):
            wavs, mels, tgts = wavs.to(rank), mels.to(rank), tgts.to(rank)

            # Discriminator
            optimizer_discriminator.zero_grad()

            wavs_ = generator(mels.squeeze(1))
            wavs_ = center_crop_waveform(wavs_, wavs.size(-1))
            mels_ = melspectrogram(wavs_)

            scores, _ = discriminator(wavs)
            scores_, _ = discriminator(wavs_.detach())

            loss_discriminator, _, _ = discriminator_loss(scores, scores_)

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Generator
            optimizer_generator.zero_grad()

            scores, features = discriminator(wavs)
            scores_, features_ = discriminator(wavs_)

            loss_mel = F.l1_loss(mels_, tgts)
            loss_features = feature_loss(features, features_)
            loss_generator_adversarial, _ = generator_loss(scores_)
            loss_generator = 45 * loss_mel + loss_features + loss_generator_adversarial

            loss_generator.backward()
            optimizer_generator.step()

            global_step += 1

            average_loss_mel += (loss_mel.item() - average_loss_mel) / i
            average_loss_discriminator += (
                loss_discriminator.item() - average_loss_discriminator
            ) / i
            average_loss_generator += (
                loss_generator.item() - average_loss_generator
            ) / i

            if rank == 0:
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar(
                        "train/loss_mel",
                        loss_mel.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_generator",
                        loss_generator.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_discriminator",
                        loss_discriminator.item(),
                        global_step,
                    )

            if global_step % args.validation_interval == 0:
                generator.eval()

                validation_loss_sum = 0.0
                validation_count = 0
                for j, (wavs, mels, tgts) in enumerate(validation_loader, 1):
                    wavs, mels, tgts = wavs.to(rank), mels.to(rank), tgts.to(rank)

                    with torch.no_grad():
                        wavs_ = generator.module(mels.squeeze(1))
                        wavs_ = center_crop_waveform(wavs_, wavs.size(-1))
                        mels_ = melspectrogram(wavs_)

                        length = min(mels_.size(-1), tgts.size(-1))

                        loss_mel = F.l1_loss(mels_[..., :length], tgts[..., :length])

                    validation_loss_sum += loss_mel.item()
                    validation_count += 1

                    if rank == 0:
                        if j <= NUM_GENERATED_EXAMPLES:
                            writer.add_audio(
                                f"generated/wav_{j}",
                                wavs_.squeeze(0),
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_figure(
                                f"generated/mel_{j}",
                                plot_spectrogram(mels_.squeeze().cpu().numpy()),
                                global_step,
                            )

                generator.train()
                discriminator.train()

                validation_stats = torch.tensor(
                    [validation_loss_sum, validation_count],
                    dtype=torch.float64,
                    device=torch.device(f"cuda:{rank}"),
                )
                dist.all_reduce(validation_stats, op=dist.ReduceOp.SUM)
                average_validation_loss = (
                    validation_stats[0] / validation_stats[1]
                ).item()
                last_validation_loss = average_validation_loss

                if rank == 0:
                    writer.add_scalar(
                        "validation/mel_loss", average_validation_loss, global_step
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, mel loss: {average_validation_loss:.4f}"
                    )

                new_best = best_loss > average_validation_loss
                if new_best:
                    best_loss = average_validation_loss

                significant_improvement = (
                    average_validation_loss
                    < early_stopping_best_metric - args.early_stopping_min_delta
                )
                if significant_improvement:
                    early_stopping_best_metric = average_validation_loss
                    bad_validations = 0
                elif global_step >= args.early_stopping_min_steps:
                    bad_validations += 1
                else:
                    bad_validations = 0

                if rank == 0 and new_best:
                    logger.info("-------- new best model found!")
                    save_checkpoint(
                        checkpoint_dir=args.checkpoint_dir,
                        generator=generator,
                        discriminator=discriminator,
                        optimizer_generator=optimizer_generator,
                        optimizer_discriminator=optimizer_discriminator,
                        scheduler_generator=scheduler_generator,
                        scheduler_discriminator=scheduler_discriminator,
                        step=global_step,
                        loss=average_validation_loss,
                        best=True,
                        latest=False,
                        bad_validations=bad_validations,
                        early_stopping_best_metric=early_stopping_best_metric,
                        best_loss=best_loss,
                        logger=logger,
                    )

                stop = (
                    args.early_stopping_patience > 0
                    and global_step >= args.early_stopping_min_steps
                    and bad_validations >= args.early_stopping_patience
                )

            if rank == 0 and global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_generator=optimizer_generator,
                    optimizer_discriminator=optimizer_discriminator,
                    scheduler_generator=scheduler_generator,
                    scheduler_discriminator=scheduler_discriminator,
                    step=global_step,
                    loss=last_validation_loss,
                    best=False,
                    latest=True,
                    bad_validations=bad_validations,
                    early_stopping_best_metric=early_stopping_best_metric,
                    best_loss=best_loss,
                    logger=logger,
                )

            if stop or (args.max_steps > 0 and global_step >= args.max_steps):
                break

        scheduler_discriminator.step()
        scheduler_generator.step()

        if rank == 0:
            logger.info(
                f"train -- epoch: {epoch}, mel loss: {average_loss_mel:.4f}, "
                f"generator loss: {average_loss_generator:.4f}, "
                f"discriminator loss: {average_loss_discriminator:.4f}"
            )
        if stop or (args.max_steps > 0 and global_step >= args.max_steps):
            break

    if rank == 0:
        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            step=global_step,
            loss=last_validation_loss,
            best=False,
            latest=True,
            bad_validations=bad_validations,
            early_stopping_best_metric=early_stopping_best_metric,
            best_loss=best_loss,
            logger=logger,
        )
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or finetune HiFi-GAN.")
    parser.add_argument(
        "--datasets",
        metavar="dataset",
        nargs="+",
        help="One or more dataset configurations in format: csv_path:wav_dir[:mel_dir]. Example: data1.csv:data/wav1:data/mel1 data2.csv:data/wav2:data/mel2",
        type=str,
        default=None
    )
    # Legacy single dataset arguments (for backward compatibility)
    parser.add_argument(
        "--csv_path",
        metavar="csv-path",
        help="[Legacy] path to the CSV file (use --datasets instead for multi-dataset)",
        type=Path
    )
    parser.add_argument(
        "--wav_dataset_dir",
        metavar="wav-dataset-dir",
        help="[Legacy] path to the wav dataset directory (use --datasets instead)",
        type=Path
    )
    parser.add_argument(
        "--mel_dataset_dir",
        metavar="mel-dataset-dir",
        help="[Legacy] path to the mel spec directory (use --datasets instead)",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory",
        type=Path
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from",
        type=Path,
    )
    parser.add_argument(
        "--finetune",
        help="deprecated alias for --reset_optim with --input_mel_source precomputed",
        action="store_true",
    )
    parser.add_argument(
        "--reset_optim",
        help="load model weights from --resume but reset optimizer, scheduler, global step, and best loss",
        action="store_true",
    )
    parser.add_argument(
        "--input_mel_source",
        help="source of generator input mel features",
        choices=["wav", "precomputed"],
        default=None,
    )
    parser.add_argument(
        "--disable_aug",
        help="disable random waveform gain and polarity augmentation during training",
        action="store_true",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="default: 2e-4 for pretraining, 1e-4 with --reset_optim",
    )
    parser.add_argument(
        "--validation_interval", type=int, default=VALIDATION_INTERVAL
    )
    parser.add_argument("--validation_subset", type=int, default=1000)
    parser.add_argument("--validation_seed", type=int, default=1234)
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    parser.add_argument("--early_stopping_min_steps", type=int, default=30000)
    args = parser.parse_args()

    # Validate arguments
    if args.datasets is None:
        if args.csv_path is None or args.wav_dataset_dir is None:
            parser.error("Either --datasets or both --csv_path and --wav_dataset_dir must be provided")

    if args.finetune and args.resume is None:
        parser.error("--finetune requires --resume")
    if args.reset_optim and args.resume is None:
        parser.error("--reset_optim requires --resume")

    if args.input_mel_source is None:
        args.input_mel_source = "precomputed" if args.finetune else "wav"
    if args.finetune:
        args.reset_optim = True
        args.disable_aug = True
    if args.learning_rate is None:
        args.learning_rate = (
            FINETUNE_LEARNING_RATE if args.reset_optim else BASE_LEARNING_RATE
        )

    if args.input_mel_source == "precomputed":
        dataset_specs = args.datasets if args.datasets is not None else [f"{args.csv_path}:{args.wav_dataset_dir}:{args.mel_dataset_dir}"]
        for dataset_spec in dataset_specs:
            if len(str(dataset_spec).split(":")) < 3:
                parser.error("input_mel_source=precomputed requires mel_dir in each dataset spec")

    # display training setup info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")

    # clear handlers
    logger.handlers.clear()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
