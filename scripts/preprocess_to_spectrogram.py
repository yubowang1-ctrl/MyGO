import os
import argparse
import numpy as np
import tensorflow as tf
import librosa
from glob import glob
from tqdm import tqdm
import sys
import concurrent.futures

# silent librosa warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from constants import *
from data.dataset import make_spectrogram

def process_file(args):
    file_path, out_dir = args
    filename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(filename)[0]
    save_path = os.path.join(out_dir, name_no_ext + ".npy")
    
    if os.path.exists(save_path):
        return

    try:
        # 1. Load Audio
        # Using librosa for m4a/wav to be consistent with dataset.py logic
        # mono=False to preserve stereo
        # sr=TARGET_SR handles resampling
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=False)
        
        # Handle dimensions: Librosa is (Channels, Time), we want (Time, Channels)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.T # (Time, Channels)
        
        # Convert to tensor
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Handle Channels (Force Stereo)
        channels = tf.shape(audio)[1]
        if channels == 1:
            audio = tf.tile(audio, [1, 2])
        elif channels > 2:
            audio = audio[:, :2]
            
        # Handle Length (Split into chunks)
        curr_len = tf.shape(audio)[0]
        chunks = []
        
        if curr_len < SAMPLE_SAMPLES:
            # Loop to fill 1 chunk
            repeats = tf.cast(tf.math.ceil(SAMPLE_SAMPLES / tf.cast(curr_len, tf.float32)), tf.int32)
            chunk = tf.tile(audio, [repeats, 1])[:SAMPLE_SAMPLES]
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            num_chunks = tf.cast(tf.math.ceil(tf.cast(curr_len, tf.float32) / SAMPLE_SAMPLES), tf.int32)
            for i in range(num_chunks):
                start = i * SAMPLE_SAMPLES
                chunk = audio[start : start + SAMPLE_SAMPLES]
                chunk_len = tf.shape(chunk)[0]
                
                # Pad/Loop the last chunk if needed
                if chunk_len < int(SAMPLE_SAMPLES * 0.1):
                    continue
                if chunk_len < SAMPLE_SAMPLES:
                    repeats = tf.cast(tf.math.ceil(SAMPLE_SAMPLES / tf.cast(chunk_len, tf.float32)), tf.int32)
                    chunk = tf.tile(chunk, [repeats, 1])[:SAMPLE_SAMPLES]
                
                chunks.append(chunk)
        
        if not chunks:
             chunks.append(tf.zeros((SAMPLE_SAMPLES, 2), dtype=tf.float32))

        # Stack chunks: (N_chunks, Time, 2)
        chunks = tf.stack(chunks)
        
        # 2. Convert to Spectrograms
        # make_spectrogram takes (Time, 2) and returns (H, W, C)
        # We pass a dummy label because make_spectrogram expects (audio, label)
        dummy_label = tf.zeros((AUDIOSET_NUM_CLASSES,))
        
        for i in range(chunks.shape[0]):
            # Run make_spectrogram
            # Note: make_spectrogram is a TF function, so we run it eagerly here
            spec, _ = make_spectrogram(chunks[i], dummy_label)
            
            # save each spec independently as a sample
            # spec is float ranged from 0 to 1, rescale to [0, 255] and save as int8
            spec = tf.clip_by_value(spec * 255.0, 0.0, 255.0)
            np.save(save_path.replace(".npy", f"_{i}.npy"), spec.numpy().astype(np.int8))
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 

def main():
    parser = argparse.ArgumentParser(description="Convert audio files to log-mel spectrograms (.npy)")
    parser.add_argument("--in_dir", required=True, help="Directory containing .m4a files")
    parser.add_argument("--out_dir", required=True, help="Directory to save .npy files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find files
    print(f"Scanning {args.in_dir}...")
    files = glob(os.path.join(args.in_dir, "*.m4a"))
    # Also look for wav if needed, but prompt specified m4a
    # files += glob(os.path.join(args.in_dir, "*.wav"))
    
    if not files:
        print("No .m4a files found.")
        return
        
    print(f"Found {len(files)} files. Processing...")
    
    # Prepare args for map
    task_args = [(f, args.out_dir) for f in files]
    
    # Use ProcessPoolExecutor
    # We force CPU for TF operations inside workers to avoid GPU OOM with multiple processes
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_file, task_args), total=len(files)))
        
    print("Done.")

if __name__ == "__main__":
    main()
