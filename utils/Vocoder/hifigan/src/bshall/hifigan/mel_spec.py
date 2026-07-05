import torch
import torchaudio
import torch.nn.functional


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=2048,
            hop_length=320,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel
