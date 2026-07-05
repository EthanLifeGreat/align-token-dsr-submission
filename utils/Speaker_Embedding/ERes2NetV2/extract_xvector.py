"""
从数据集中提取 eres2netv2 x-vector
使用 iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common 模型
输出结构：xvector/eresnet2v2/{spk_id}/{wav_id}.npy

支持任何符合 wav/{spk_id}/{wav_id}.wav 结构的数据集
即使 spk_id 或 wav_id 中包含斜杠也能正确处理

用法:
    python extract_xvector.py --dataset <dataset_name> --device <gpu_id>
    
示例:
    python extract_xvector.py --dataset MAGICDATA-cut --device cuda:0
    python extract_xvector.py --dataset LibriTTS-16k --device cuda:1
    python extract_xvector.py --dataset THCHS30 --device cuda:2
"""

import os
import sys
import csv
import pathlib
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import warnings
import logging

speakerlab_path = os.environ.get("SPEAKERLAB_PATH")
if speakerlab_path:
    sys.path.insert(0, speakerlab_path)

try:
    from speakerlab.process.processor import FBank
except ImportError as e:
    print(f"警告：无法导入 speakerlab: {e}")
    FBank = None

from modelscope.hub.snapshot_download import snapshot_download


class SpeakerVerification:
    """说话人验证类"""

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
                 device: str = 'auto', verbose: bool = True):
        self.model_id = model_id
        self.local_model_dir = pathlib.Path(local_model_dir)
        self.device = self._setup_device(device)
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

        self.conf = self.SUPPORTS[model_id]
        self.model = None
        self.feature_extractor = None
        self._load_model()

    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_model(self):
        save_dir = self.local_model_dir / self.model_id.split('/')[1]
        save_dir.mkdir(exist_ok=True, parents=True)

        model_path = save_dir / self.conf['model_pt']
        if not model_path.exists():
            if self.verbose:
                print(f"下载模型 {self.model_id}...")
            cache_dir = snapshot_download(self.model_id, revision=self.conf['revision'])
            cache_dir = pathlib.Path(cache_dir)

            for src in cache_dir.glob('*'):
                if self.conf['model_pt'] in src.name:
                    dst = save_dir / src.name
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.absolute())
                    if self.verbose:
                        print(f"已链接模型文件：{dst} -> {src}")

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
            raise ImportError("无法初始化 FBank 特征提取器")

        if self.verbose:
            print(f"模型加载完成，使用设备：{self.device}")
            print(f"模型维度：{self.conf['model_config']['args'].get('embedding_size', '未知')}")

    def _dynamic_import(self, import_path: str):
        module_path, class_name = import_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _load_wav(self, wav_file: str, target_sample_rate: int = 16000) -> torch.Tensor:
        try:
            wav, fs = torchaudio.load(wav_file)
            if wav.numel() == 0 or wav.shape[1] == 0:
                raise ValueError(f"音频文件 {wav_file} 为空或无法读取")
            if fs != target_sample_rate:
                wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=target_sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav
        except Exception as e:
            raise RuntimeError(f"加载音频文件 {wav_file} 失败：{str(e)}")

    def extract_embedding(self, wav_file: str, normalize: bool = True) -> np.ndarray:
        if self.model is None or self.feature_extractor is None:
            raise RuntimeError("模型未正确初始化")

        wav = self._load_wav(wav_file)

        if wav.shape[1] < 1600:
            warnings.warn(f"音频 {wav_file} 过短 ({wav.shape[1] / 16000:.2f}秒)")

        try:
            feat = self.feature_extractor(wav)
            if feat.shape[0] < 10:
                raise ValueError(f"音频 {wav_file} 特征帧数不足 ({feat.shape[0]}帧)")
            feat = feat.unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"提取特征失败 {wav_file}: {str(e)}")

        with torch.no_grad():
            try:
                embedding = self.model(feat)
                if embedding.dim() > 2:
                    embedding = embedding.squeeze(0)
                embedding = embedding.detach().cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"���型推理失败 {wav_file}: {str(e)}")

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding


def main():
    parser = argparse.ArgumentParser(description='从数据集中提取 x-vector')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称（如 MAGICDATA-cut, LibriTTS-16k, THCHS30）')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Dataset root containing meta.csv and wav/')
    parser.add_argument('--device', type=str, default='auto',
                        help='GPU 设备（如 cuda:0, cuda:1, auto）')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='是否显示详细信息')
    args = parser.parse_args()

    # 配置路径
    data_root = pathlib.Path(args.data_root)
    meta_csv = data_root / 'meta.csv'
    wav_dir = data_root / 'wav'
    xvec_dir = data_root / 'xvector' / 'eresnet2v2'

    if not meta_csv.exists():
        print(f"错误：找不到 meta.csv: {meta_csv}")
        sys.exit(1)

    if not wav_dir.exists():
        print(f"错误：找不到 wav 目录：{wav_dir}")
        sys.exit(1)

    # 创建输出目录
    xvec_dir.mkdir(parents=True, exist_ok=True)

    # 读取 meta.csv 获取 wav 文件列表
    wav_files = []
    wav_ids = []
    spk_ids = []
    with open(meta_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            spk_id = row['spk_id']
            wav_id = row['wav_id']
            # 构建 wav 路径：wav/{spk_id}/{wav_id}.wav
            # 即使 spk_id 或 wav_id 包含斜杠也能正确处理
            wav_path = wav_dir / spk_id / f'{wav_id}.wav'
            if wav_path.exists():
                wav_files.append(str(wav_path))
                wav_ids.append(wav_id)
                spk_ids.append(spk_id)

    print(f"找到 {len(wav_files)} 个 wav 文件")

    # 初始化模型
    model_id = 'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common'
    print(f"初始化模型：{model_id}")
    sv = SpeakerVerification(model_id=model_id, device=args.device, verbose=args.verbose)

    # 批量提取 x-vector
    success_count = 0
    fail_count = 0
    skip_count = 0

    for wav_path, wav_id, spk_id in tqdm(zip(wav_files, wav_ids, spk_ids), desc="提取 x-vector"):
        # 按 spk_id/wav_id.npy 结构保存
        # 即使 spk_id 包含斜杠（如 train/14_3466），也能正确创建目录
        spk_xvec_dir = xvec_dir / spk_id
        spk_xvec_dir.mkdir(parents=True, exist_ok=True)
        xvec_path = spk_xvec_dir / f'{wav_id}.npy'

        # 跳过已存在的文件（断点续传）
        if xvec_path.exists():
            skip_count += 1
            continue

        try:
            embedding = sv.extract_embedding(wav_path, normalize=True)
            np.save(xvec_path, embedding)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"\n处理失败 {wav_path}: {e}")

    print(f"\n完成！成功：{success_count}, 失败：{fail_count}, 跳过：{skip_count}")


if __name__ == '__main__':
    main()
