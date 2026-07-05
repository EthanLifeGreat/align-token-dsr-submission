"""
推理脚本
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import numpy as np

from src.wav2phoneme.config import ModelConfig, load_model_config
from src.wav2phoneme.model import Wav2PhonemeModel
from src.wav2phoneme.utils import load_phone_dict, load_id_to_phone


class Wav2PhonemeInference:
    """
    Wav2Phoneme 推理类
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        phone_dict_path: str = "data/AISHELL-2/phone_dict.json",
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_path: 模型 checkpoint 路径
            phone_dict_path: phone_dict.json 路径
            device: 设备 ("cuda" 或 "cpu")
        """
        self.device = device
        
        # 加载模型
        checkpoint_path = Path(checkpoint_path)
        if (checkpoint_path / "config.yaml").exists():
            config = load_model_config(checkpoint_path / "config.yaml")
        else:
            # 使用默认配置
            config = ModelConfig()
        
        self.model = Wav2PhonemeModel.from_pretrained(str(checkpoint_path))
        self.model.to(device)
        self.model.eval()
        
        # 加载音素字典
        self.phone_dict = load_phone_dict(phone_dict_path)
        self.id_to_phone = load_id_to_phone(phone_dict_path)
    
    def load_audio(self, audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            sample_rate: 目标采样率
        
        Returns:
            audio: [samples]
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0).numpy()
    
    @torch.no_grad()
    def infer(
        self,
        audio: Union[str, np.ndarray],
        return_logits: bool = False,
    ) -> Dict:
        """
        推理
        
        Args:
            audio: 音频文件路径或音频数组
            return_logits: 是否返回 logits
        
        Returns:
            {
                "phoneme_ids": List[int],
                "phoneme_names": List[str],
                "logits": np.ndarray (optional),
            }
        """
        # 加载音频
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(str(audio))
        
        # 转为 tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(audio_tensor, dtype=torch.long)
        
        # 推理
        if return_logits:
            phoneme_ids, logits = self.model.generate(
                audio_tensor, 
                attention_mask, 
                return_logits=True
            )
            logits = logits.cpu().numpy()
        else:
            phoneme_ids = self.model.generate(audio_tensor, attention_mask)
        
        # 转为列表
        phoneme_ids = phoneme_ids[0].cpu().numpy().tolist()
        
        # 转为音素名称
        phoneme_names = [self.id_to_phone.get(pid, "unk") for pid in phoneme_ids]
        
        result = {
            "phoneme_ids": phoneme_ids,
            "phoneme_names": phoneme_names,
        }
        
        if return_logits:
            result["logits"] = logits
        
        return result
    
    def infer_batch(
        self,
        audio_paths: List[str],
        return_logits: bool = False,
    ) -> List[Dict]:
        """
        批量推理
        
        Args:
            audio_paths: 音频文件路径列表
            return_logits: 是否返回 logits
        
        Returns:
            List of results
        """
        results = []
        for audio_path in audio_paths:
            result = self.infer(audio_path, return_logits)
            result["audio_path"] = audio_path
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Wav2Phoneme Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file or directory"
    )
    parser.add_argument(
        "--phone-dict",
        type=str,
        default="data/AISHELL-2/phone_dict.json",
        help="Path to phone_dict.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--return-logits",
        action="store_true",
        help="Return logits"
    )
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = Wav2PhonemeInference(
        checkpoint_path=args.checkpoint,
        phone_dict_path=args.phone_dict,
        device=args.device,
    )
    
    # 推理
    audio_path = Path(args.audio)
    
    if audio_path.is_file():
        # 单文件
        result = inferencer.infer(str(audio_path), args.return_logits)
        results = [{"audio_path": str(audio_path), **result}]
    else:
        # 目录
        audio_files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.WAV"))
        results = inferencer.infer_batch(
            [str(f) for f in audio_files],
            args.return_logits
        )
    
    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
