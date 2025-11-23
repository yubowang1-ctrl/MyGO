#!/usr/bin/env python3
"""
Batch Audio Preprocessor for TF (Optimized) - Final Log Version
将音频文件夹批量转换为 H*W*2 的 .npy 频谱图。

Changes:
- 强制执行 Log Magnitude 处理 (np.log(S + eps))。
- 生成 Global (100%) 和 Local (30%) 两种视图。
"""

import argparse
import sys
import os
import math
from pathlib import Path
import logging
import random

import tensorflow as tf
import librosa
import numpy as np
from tqdm import tqdm

from utils.augment_utils import SpectrogramAugmentor
from utils.audio_process import AudioPreprocessor, AudioSTFTConfig


# 配置日志
logging.basicConfig(
    filename='preprocess_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === 常量配置 ===
TARGET_SR = 48000
MAX_DURATION_PER_CHUNK = 180.0  # 每块最多处理 180秒 (3分钟)，防止显存爆炸
EPSILON = 1e-5  # 用于 Log 运算防止 log(0)

def setup_gpu():
    """配置 GPU 显存按需分配"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(e)

def create_local_view(global_spec, ratio=0.3):
    """
    输入: Global View [H, W, 2]
    输出: Local View [H, W*0.3, 2]
    逻辑: 保留所有频率，随机切取 30% 的时间片段。
    """
    h, w, c = global_spec.shape
    crop_w = int(w * ratio)
    
    # 边界保护
    if crop_w < 1: crop_w = 1
    if w <= crop_w: return global_spec.copy()
    
    # 随机选择起始点
    max_start = w - crop_w
    start = random.randint(0, max_start)
    end = start + crop_w
    
    # 切片: [:, 时间, :]
    local_spec = global_spec[:, start:end, :]
    return local_spec

def load_audio_raw(path):
    """只负责快速读取 IO，不做重采样"""
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1: y = y[np.newaxis, :]
    elif y.ndim == 2 and y.shape[0] > y.shape[1]: y = y.T
    return y.astype(np.float32), sr

def process_waveform_in_chunks(preprocessor, waveform, native_sr, chunk_seconds=MAX_DURATION_PER_CHUNK, **stft_kwargs):
    """内存保护逻辑：长波形切片 -> GPU -> 拼接"""
    n_channels, n_samples = waveform.shape
    samples_per_chunk = int(chunk_seconds * native_sr)
    
    if n_samples <= samples_per_chunk:
        return _run_tf_inference(preprocessor, waveform, native_sr, **stft_kwargs)

    num_chunks = math.ceil(n_samples / samples_per_chunk)
    spec_chunks = []
    
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = min((i + 1) * samples_per_chunk, n_samples)
        sub_wav = waveform[:, start:end]
        sub_spec = _run_tf_inference(preprocessor, sub_wav, native_sr, **stft_kwargs)
        spec_chunks.append(sub_spec)
        
    return np.concatenate(spec_chunks, axis=1)

def _run_tf_inference(preprocessor, waveform_np, sr_int, **stft_kwargs):
    """单次 TF 推理封装"""
    wav_tf = tf.convert_to_tensor(waveform_np, dtype=tf.float32)
    sr_tf = tf.constant(sr_int, dtype=tf.int32)
    spec = preprocessor(wav_tf, sr_tf, **stft_kwargs)
    if hasattr(spec, 'numpy'): spec = spec.numpy()
    else: spec = np.array(spec)
    return spec

def enforce_stereo_shape(spec_np):
    """强制输出形状为 (H, W, 2)"""
    if spec_np.ndim == 2: spec_np = spec_np[:, :, np.newaxis]
    h, w, c = spec_np.shape
    
    is_fake = False
    if c == 1:
        spec_np = np.concatenate([spec_np, spec_np], axis=2)
        is_fake = True
    elif c > 2:
        spec_np = spec_np[:, :, :2]
    
    return spec_np, is_fake

def main():
    parser = argparse.ArgumentParser(description="Optimized Batch Audio Preprocessor")
    parser.add_argument("--input_dir", required=True, help="Input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    # 注意：--log_mag 参数已被删除，因为现在是强制执行
    
    parser.add_argument("--fmin", type=float, default=30.0)
    parser.add_argument("--fmax", type=float, default=16000.0)
    
    args = parser.parse_args()
    setup_gpu()

    in_path = Path(args.input_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Initializing TensorFlow AudioPreprocessor...")
    cfg = AudioSTFTConfig(target_sr=TARGET_SR)
    preprocessor = AudioPreprocessor(cfg)
    
    files = []
    for ext in ['*.m4a', '*.wav', '*.flac', '*.mp3']:
        files.extend(list(in_path.rglob(ext)))
    
    if not files:
        print("No audio files found.")
        sys.exit(0)
        
    print(f"Found {len(files)} files. Processing...")

    success_count = 0
    stft_params = {'fmin': args.fmin, 'fmax': args.fmax}

    # 设置随机种子以保证可复现性
    random.seed(42)
    augmentor = SpectrogramAugmentor()

    for file_p in tqdm(files, desc="Processing"):
        try:
            # 1. 加载 & 裁剪声道
            waveform, native_sr = load_audio_raw(str(file_p))
            if waveform.shape[0] > 2: waveform = waveform[:2, :]
            
            # 2. STFT 生成 (Linear Magnitude)
            spec_np = process_waveform_in_chunks(preprocessor, waveform, native_sr, **stft_params)
            
            # 3. 强制立体声格式
            spec_global, is_fake_stereo = enforce_stereo_shape(spec_np)
            
            # 4. 【关键修改】强制 Log 处理 (对数幅度谱)
            #    对矩阵中每个数值进行 log(x + eps)
            spec_global = np.log(spec_global + EPSILON)

            # 5. 生成 Local View (在 Log 之后切，保证分布一致)
            view_tensors = augmentor.generate_views(spec_global, global_view_num=1, local_view_num=1)

            # 保存 (需要先转回 numpy 如果你要存 .npy)
            global_aug = view_tensors[0].numpy()
            local_aug = view_tensors[1].numpy()

            np.save(out_path / f"{file_p.stem}_global_aug.npy", global_aug)
            np.save(out_path / f"{file_p.stem}_local_aug.npy", local_aug)
            
            status_msg = f"Processed {file_p.name} -> Global:{spec_global.shape} Local:{spec_local.shape}"
            logging.info(status_msg)
            success_count += 1

        except Exception as e:
            err_msg = f"Failed {file_p.name}: {str(e)}"
            logging.error(err_msg)
            continue

    print(f"\nDone! Processed {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()