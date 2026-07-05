#!/usr/bin/env python3
"""
Script to extract generator weights from a training checkpoint.
This script extracts only the generator weights from a full training checkpoint
and saves them in a smaller file for inference.

Usage:
    python extract_weights.py <input_checkpoint> [--output <output_path>]

Example:
    python extract_weights.py /path/to/model-best.pt --output ckpt/generator.pt
"""
import argparse
from pathlib import Path
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def extract_generator_weights(
    input_path: str,
    output_path: str,
) -> None:
    """Extract generator weights from a training checkpoint.

    Args:
        input_path: Path to the full training checkpoint.
        output_path: Path to save the extracted generator weights.
    """
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Extract generator state dict
    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume the checkpoint is already just the state dict
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")

    # Save only the generator weights
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving generator weights to {output_path}")
    torch.save(state_dict, output_path)

    # Print size comparison
    input_size = Path(input_path).stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Original checkpoint size: {input_size:.2f} MB")
    print(f"Generator weights size: {output_size:.2f} MB")
    print(f"Size reduction: {(1 - output_size/input_size)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Extract generator weights from HiFiGAN training checkpoint"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input checkpoint file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the extracted weights (default: ckpt/generator.pt in same directory as this script)",
    )

    args = parser.parse_args()

    if args.output is None:
        output_path = Path(__file__).parent / "ckpt" / "generator.pt"
    else:
        output_path = Path(args.output)

    extract_generator_weights(args.input, str(output_path))


if __name__ == "__main__":
    main()
