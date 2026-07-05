"""
HiFiGAN Inference API
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from .model import HifiganGenerator


class HiFiGANAPI:
    """HiFiGAN Vocoder API for mel-spectrogram to waveform conversion.

    This API provides a simple interface to convert mel-spectrograms to waveforms
    using a pre-trained HiFiGAN model.

    Example:
        >>> from utils.Vocoder.hifigan import HiFiGANAPI
        >>> vocoder = HiFiGANAPI()  # Uses default checkpoint
        >>> mel = np.random.randn(100, 80)  # 100 frames, 80 mel bins
        >>> wav, sr = vocoder.inference(mel)
        >>> print(f"Generated waveform shape: {wav.shape}, sample rate: {sr}")
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        remove_weight_norm: bool = True,
    ):
        """Initialize HiFiGAN vocoder.

        Args:
            checkpoint_path: Path to the model checkpoint file. If None, uses the default
                           checkpoint at 'ckpt/generator.pt' relative to this file's directory.
            device: Device to run the model on. If None, automatically selects 'cuda' if available,
                   otherwise 'cpu'.
            remove_weight_norm: Whether to remove weight normalization after loading.
                              This improves inference speed.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set checkpoint path
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / "ckpt" / "generator.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint to detect mel_bins
        state_dict = self._load_state_dict(checkpoint_path)

        # Detect mel_bins from conv_pre weight shape
        # Check for weight_norm format (weight_g/weight_v) or normal format (weight)
        if "conv_pre.weight" in state_dict:
            conv_pre_weight = state_dict["conv_pre.weight"]
            self._mel_bins = conv_pre_weight.shape[1]
            self._has_weight_norm = False
        elif "conv_pre.weight_g" in state_dict:
            # weight_norm format: weight_g shape is (out_channels, 1, 1), weight_v shape is (out, in, k)
            conv_pre_weight_v = state_dict["conv_pre.weight_v"]
            self._mel_bins = conv_pre_weight_v.shape[1]
            self._has_weight_norm = True
        else:
            raise ValueError("Cannot find 'conv_pre.weight' or 'conv_pre.weight_g' in checkpoint. Invalid checkpoint format.")

        print(f"[HiFiGAN] Model expects {self._mel_bins} mel bins as input.")

        # Load model with detected mel_bins
        self.model = HifiganGenerator(in_channels=self._mel_bins).to(self.device)

        # Load state dict
        if self._has_weight_norm:
            # For weight_norm format, load directly then remove weight norm
            self.model.load_state_dict(state_dict)
        else:
            # For normal format, load directly
            self.model.load_state_dict(state_dict)

        # Remove weight norm for faster inference
        if remove_weight_norm:
            self.model.remove_weight_norm()

        self.model.eval()

    def _load_state_dict(self, checkpoint_path: Path) -> dict:
        """Load state dict from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            State dict of the generator model.
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "generator" in checkpoint:
            # Full training checkpoint with optimizer states
            state_dict = checkpoint["generator"]["model"]
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
        elif "state_dict" in checkpoint:
            # Checkpoint with 'state_dict' key
            state_dict = checkpoint["state_dict"]
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
        else:
            # Assume it's just the model state dict
            state_dict = checkpoint
            consume_prefix_in_state_dict_if_present(state_dict, "module.")

        print("Checkpoint loaded successfully")
        return state_dict

    @property
    def mel_bins(self) -> int:
        """Return the number of mel bins expected by the model."""
        return self._mel_bins

    @property
    def sample_rate(self) -> int:
        """Return the sample rate of the generated audio."""
        return self.model.sample_rate

    @torch.no_grad()
    def inference(
        self,
        mel: np.ndarray,
        output_numpy: bool = True,
    ) -> Union[Tuple[np.ndarray, int], Tuple[torch.Tensor, int]]:
        """Convert mel-spectrogram to waveform.

        Args:
            mel: Input mel-spectrogram as numpy array of shape (time, mel_bins) or
                (mel_bins, time). The model expects mel_bins as the first dimension
                after batch, so shape (mel_bins, time) is preferred for efficiency.
            output_numpy: If True, returns numpy array. If False, returns torch tensor.

        Returns:
            Tuple of (waveform, sample_rate):
                - waveform: Generated waveform as numpy array of shape (time,) or torch tensor
                - sample_rate: Sample rate of the generated audio (default: 16000)

        Example:
            >>> mel = np.random.randn(100, 80)  # 100 frames, 80 mel bins
            >>> wav, sr = vocoder.inference(mel)
        """
        # Handle input shape
        if mel.ndim != 2:
            raise ValueError(f"Expected 2D mel-spectrogram, got shape {mel.shape}")

        # Check and transpose if needed
        # Model expects (mel_bins, time), so we need to check if last dim matches mel_bins
        if mel.shape[-1] == self._mel_bins:
            # Shape is (time, mel_bins), need transpose
            mel = mel.T
        elif mel.shape[0] != self._mel_bins:
            raise ValueError(
                f"Input mel-spectrogram shape {mel.shape} doesn't match expected mel_bins={self._mel_bins}. "
                f"Expected shape (time, {self._mel_bins}) or ({self._mel_bins}, time)."
            )
        # else: shape is already (mel_bins, time), no transpose needed

        # Convert to tensor and add batch dimension
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)

        # Generate waveform
        # Input shape: (batch, mel_bins, time)
        # Output shape: (batch, 1, time)
        wav_tensor = self.model(mel_tensor)

        # Remove batch and channel dimensions
        wav_tensor = wav_tensor.squeeze()

        if output_numpy:
            return wav_tensor.cpu().numpy(), self.sample_rate
        else:
            return wav_tensor, self.sample_rate

    @torch.no_grad()
    def inference_batch(
        self,
        mels: np.ndarray,
        output_numpy: bool = True,
    ) -> Union[Tuple[np.ndarray, int], Tuple[torch.Tensor, int]]:
        """Convert batch of mel-spectrograms to waveforms.

        Args:
            mels: Input mel-spectrograms as numpy array of shape (batch, mel_bins, time) or
                 (batch, time, mel_bins). The model expects mel_bins as the second dimension,
                 so shape (batch, mel_bins, time) is preferred for efficiency.
            output_numpy: If True, returns numpy array. If False, returns torch tensor.

        Returns:
            Tuple of (waveforms, sample_rate):
                - waveforms: Generated waveforms as numpy array of shape (batch, time) or torch tensor
                - sample_rate: Sample rate of the generated audio
        """
        if mels.ndim != 3:
            raise ValueError(f"Expected 3D batch of mel-spectrograms, got shape {mels.shape}")

        # Check and transpose if needed
        # Model expects (batch, mel_bins, time), so we need to check if last dim matches mel_bins
        if mels.shape[-1] == self._mel_bins:
            # Shape is (batch, time, mel_bins), need transpose
            mels = np.transpose(mels, (0, 2, 1))
        elif mels.shape[1] != self._mel_bins:
            raise ValueError(
                f"Input mel-spectrograms shape {mels.shape} doesn't match expected mel_bins={self._mel_bins}. "
                f"Expected shape (batch, time, {self._mel_bins}) or (batch, {self._mel_bins}, time)."
            )
        # else: shape is already (batch, mel_bins, time), no transpose needed

        # Convert to tensor
        mel_tensor = torch.FloatTensor(mels).to(self.device)

        # Generate waveforms
        wav_tensor = self.model(mel_tensor)

        # Remove channel dimension: (batch, 1, time) -> (batch, time)
        wav_tensor = wav_tensor.squeeze(1)

        if output_numpy:
            return wav_tensor.cpu().numpy(), self.sample_rate
        else:
            return wav_tensor, self.sample_rate


def load_vocoder(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> HiFiGANAPI:
    """Convenience function to load the HiFiGAN vocoder.

    Args:
        checkpoint_path: Path to the model checkpoint. If None, uses default.
        device: Device to run on. If None, auto-selects.

    Returns:
        HiFiGANAPI instance
    """
    return HiFiGANAPI(checkpoint_path=checkpoint_path, device=device)
