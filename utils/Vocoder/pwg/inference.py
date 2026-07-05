"""Parallel WaveGAN inference API."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import yaml

from .model import ParallelWaveGANGenerator


class ParallelWaveGANAPI:
    """Convert 2-D mel-spectrograms to waveforms with Parallel WaveGAN."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        remove_weight_norm: bool = True,
    ):
        root = Path(__file__).parent
        self.checkpoint_path = Path(
            checkpoint_path
            or root / "ckpt" / "checkpoint-400000steps.pkl"
        )
        self.config_path = Path(config_path or root / "config.yml")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"PWG checkpoint not found: {self.checkpoint_path}"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"PWG config not found: {self.config_path}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        params = self.config["generator_params"]
        self._mel_bins = int(params["aux_channels"])
        self._sample_rate = int(self.config["sampling_rate"])
        self.model = ParallelWaveGANGenerator(**params)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"]["generator"])
        if remove_weight_norm:
            self.model.remove_weight_norm()
        self.model.eval().to(self.device)

    @property
    def mel_bins(self) -> int:
        return self._mel_bins

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _prepare_mel(self, mel: np.ndarray) -> torch.Tensor:
        mel = np.asarray(mel)
        if mel.ndim != 2:
            raise ValueError(
                f"Expected a 2-D mel-spectrogram, got shape {mel.shape}"
            )
        if mel.shape[-1] == self.mel_bins:
            mel = mel
        elif mel.shape[0] == self.mel_bins:
            mel = mel.T
        else:
            raise ValueError(
                f"Mel shape {mel.shape} does not contain "
                f"{self.mel_bins} mel bins"
            )
        return torch.as_tensor(
            np.ascontiguousarray(mel), dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def inference(
        self,
        mel: np.ndarray,
        output_numpy: bool = True,
    ) -> Union[Tuple[np.ndarray, int], Tuple[torch.Tensor, int]]:
        conditioning = self._prepare_mel(mel)
        frame_count = conditioning.size(0)
        conditioning = conditioning.T.unsqueeze(0)
        conditioning = torch.nn.functional.pad(
            conditioning,
            (self.model.aux_context_window,) * 2,
            mode="replicate",
        )
        noise = torch.randn(
            1,
            1,
            frame_count * self.model.upsample_factor,
            device=self.device,
        )
        waveform = self.model(noise, conditioning).reshape(-1)
        if output_numpy:
            waveform = waveform.cpu().numpy()
        return waveform, self.sample_rate


def load_vocoder(
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ParallelWaveGANAPI:
    return ParallelWaveGANAPI(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

