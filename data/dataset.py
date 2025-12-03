import tensorflow as tf
import librosa
import numpy as np
import os

# ==============================================================================
# 1. Configuration & Hyperparameters
# ==============================================================================
# Audio Config
TARGET_SR = 48000
DURATION = 10.0 # seconds
SAMPLE_SAMPLES = int(TARGET_SR * DURATION) # 480,000 samples

# Spectrogram Config
N_FFT = 4096 # the frequency resolution is TARGET_SR / N_FFT = ~11.7 Hz
HOP_LEN = 2048 # int(0.1 * TARGET_SR) # 0.1 seconds hop
N_MELS = 256
FMIN = 60.0
FMAX = 12000.0
EPSILON = 1e-6

# View Generation Config
NUM_GLOBAL_VIEWS = 2
NUM_LOCAL_VIEWS = 6
TOTAL_VIEWS = NUM_GLOBAL_VIEWS + NUM_LOCAL_VIEWS

# Image Output Config
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 2 # Stereo

# ==============================================================================
# 2. Audio Loading (Python Function)
# ==============================================================================
def python_load_m4a(file_path_tensor):
    """
    Loads audio using librosa (ffmpeg backend), handles resampling, 
    stereo enforcement, and splits into 10s chunks.
    """
    file_path = file_path_tensor.numpy().decode('utf-8')
    
    try:
        # Load with librosa (handles resampling automatically)
        # mono=False preserves channels
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=False)
        
        # Handle dimensions: Librosa is (Channels, Time), we want (Time, Channels)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.T
        
        # Handle Channels (Force Stereo)
        if audio.shape[1] == 1:
            audio = np.tile(audio, (1, 2))
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
            
        # Handle Length (Split into chunks)
        curr_len = audio.shape[0]
        chunks = []
        
        if curr_len < SAMPLE_SAMPLES:
            # Loop to fill 1 chunk
            repeats = int(np.ceil(SAMPLE_SAMPLES / curr_len))
            chunk = np.tile(audio, (repeats, 1))[:SAMPLE_SAMPLES]
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            num_chunks = int(np.ceil(curr_len / SAMPLE_SAMPLES))
            for i in range(num_chunks):
                start = i * SAMPLE_SAMPLES
                end = start + SAMPLE_SAMPLES
                chunk = audio[start:end]
                
                # Pad/Loop the last chunk if needed
                # if the length of last chunk is less than 1s, discard
                if chunk.shape[0] < SAMPLE_SAMPLES * 0.1:
                    continue
                if chunk.shape[0] < SAMPLE_SAMPLES:
                    repeats = int(np.ceil(SAMPLE_SAMPLES / chunk.shape[0]))
                    chunk = np.tile(chunk, (repeats, 1))[:SAMPLE_SAMPLES]
                
                chunks.append(chunk)
            
        return np.stack(chunks).astype(np.float32)
        
    except Exception as e:
        # Return 1 silent chunk on error
        print(f"Error loading {file_path}: {e}")
        return np.zeros((1, SAMPLE_SAMPLES, 2), dtype=np.float32)

def load_audio_dataset(file_path):
    """
    Wraps the python function and returns a Dataset of chunks.
    """
    [audio_chunks] = tf.py_function(
        func=python_load_m4a,
        inp=[file_path],
        Tout=[tf.float32]
    )
    # Set shape: (None, SAMPLE_SAMPLES, 2)
    audio_chunks.set_shape([None, SAMPLE_SAMPLES, 2])
    
    return tf.data.Dataset.from_tensor_slices(audio_chunks)

# ==============================================================================
# 3. Spectrogram Generation (TF Graph)
# ==============================================================================
def make_spectrogram(audio):
    """
    Convert waveform to Log-Mel Spectrogram and Normalize.
    Input: (Time, 2)
    Output: (H, W, 2) -> (N_MELS, TimeSteps, 2)
    """
    # Transpose to (Channels, Time) for stft
    audio_t = tf.transpose(audio) 
    
    # 1. STFT
    stft = tf.signal.stft(
        audio_t, 
        frame_length=N_FFT, 
        frame_step=HOP_LEN,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
    )
    magnitudes = tf.abs(stft) # shape: (Channels, Time, Freq)
    
    # 2. Mel Scale
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        N_MELS, num_spectrogram_bins, TARGET_SR, FMIN, FMAX
    )
    
    # (Channels, Time, Freq) dot (Freq, Mels) -> (Channels, Time, Mels)
    mel_spectrograms = tf.tensordot(magnitudes, linear_to_mel_weight_matrix, 1)
    
    # 3. Log Magnitude (Decibels)
    # log(S + eps)
    log_mel = tf.math.log(mel_spectrograms + EPSILON)
    
    # 4. Normalization (Min-Max Scaling to 0~1)
    # Scale to [0, 1]
    # log_mel = (log_mel - tf.reduce_min(log_mel)) / (tf.reduce_max(log_mel) - tf.reduce_min(log_mel) + EPSILON)
    tf.print("log_mel stats:", tf.reduce_max(log_mel), tf.reduce_min(log_mel), tf.reduce_mean(log_mel))
    # 5. Reshape for ViT
    # Current: (Channels, Time, Mels)
    # Target: (Mels, Time, Channels) -> (H, W, C)
    log_mel = tf.transpose(log_mel, perm=[2, 1, 0])
    
    # 6. Resize to fixed input size for ViT (224x224)
    # Note: This distorts time/freq aspect ratio
    # log_mel = tf.image.resize(log_mel, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    return log_mel

# ==============================================================================
# 4. View Generation (Augmentation)
# ==============================================================================
def generate_views(spectrogram):
    """
    Takes one full spectrogram and generates V views.
    Input: (H, W, C)
    Output: (V, H, W, C)
    """
    views = []
    
    # Placeholder for Data Augmentation
    # Ideally, you would use random crops, masking, etc.
    
    # 1. Global Views
    for _ in range(NUM_GLOBAL_VIEWS):
        # Just duplicate for now
        views.append(spectrogram)
        
    # 2. Local Views
    for _ in range(NUM_LOCAL_VIEWS):
        # Just duplicate for now
        views.append(spectrogram)
        
    return tf.stack(views)

# ==============================================================================
# 5. Dataset Pipeline
# ==============================================================================
def get_dataset(data_dir, batch_size):
    """
    Creates the full training dataset pipeline.
    """
    # 1. List files (Lazy)
    # Supports both .wav and .m4a if using the python loader
    file_pattern = str(data_dir) + "/*.m4a" 
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    
    # 2. Load Audio (Parallel Python - CPU)
    # Use interleave to flatten chunks from each file
    ds = ds.interleave(
        load_audio_dataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # 3. Convert to Spectrogram (Native TF - GPU/CPU)
    ds = ds.map(make_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Generate Views (Native TF - GPU/CPU)
    ds = ds.map(generate_views, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 5. Batch and Prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def visualize_sample(data_dir):
    """
    Takes one sample from the dataset and plots the generated spectrograms (views).
    """
    import matplotlib.pyplot as plt
    
    # Create a small dataset with batch_size=1
    ds = get_dataset(data_dir, batch_size=1)
    
    # Take 1 batch
    for batch in ds.take(1):
        # batch shape: (1, TOTAL_VIEWS, H, W, C)
        views = batch[0] # (TOTAL_VIEWS, H, W, C)
        
        num_views = views.shape[0]
        
        # Plot
        fig, axes = plt.subplots(2, (num_views + 1) // 2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(num_views):
            # Get the i-th view
            # Shape: (H, W, C) -> We visualize the first channel (Left)
            spec = views[i, :, :, 0].numpy()
            
            # Flip Y axis so low freq is at bottom
            spec = np.flipud(spec)
            
            ax = axes[i]
            im = ax.imshow(spec, aspect='auto')
            ax.set_title(f"View {i} {'(Global)' if i < NUM_GLOBAL_VIEWS else '(Local)'}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        break

visualize_sample("/Users/elly/Desktop/T7/test")