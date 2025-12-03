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
    log_mel = (log_mel - tf.reduce_min(log_mel)) / (tf.reduce_max(log_mel) - tf.reduce_min(log_mel) + EPSILON)
    # tf.print("log_mel stats:", tf.reduce_max(log_mel), tf.reduce_min(log_mel), tf.reduce_mean(log_mel))
    
    # 5. Reshape for ViT
    # Current: (Channels, Time, Mels)
    # Target: (Mels, Time, Channels) -> (H, W, C)
    log_mel = tf.transpose(log_mel, perm=[2, 1, 0])
    
    # 6. Resize to fixed input size for ViT (224x224)
    # Note: This distorts time/freq aspect ratio
    # log_mel = tf.image.resize(log_mel, [IMAGE_HEIGHT, IMAGE_WIDTH])
    tf.print("Before resize:", tf.shape(log_mel))
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
    
    # 1. Global Views
    for _ in range(NUM_GLOBAL_VIEWS):
        masked = random_mask(spectrogram, num_masks=2, mask_size=30, fill_value=0.0)
        shifted = random_pitch_shift(masked, max_shift=2)
        views.append(shifted)
        
    # 2. Local Views
    for _ in range(NUM_LOCAL_VIEWS):
        cropped = random_time_crop(spectrogram, percent=0.3)
        dilated = random_time_dilation(cropped, scale_range=(0.7, 1.3))
        shifted = random_pitch_shift(dilated, max_shift=5)
        conved = random_conv2d(shifted, magnitude=0.1)
        
        masked = random_mask(conved, num_masks=20, mask_size=30, fill_value=0.0)
        dropped = random_drop_channels(masked, prob=0.05)
        views.append(dropped)
        
    return tf.stack(views)

def random_mask(spec, num_masks=1, mask_size=20, fill_value=0.0):
    """
    Applies random time-frequency masks to the spectrogram.
    spec: (H, W, C)
    num_masks: number of masks to apply
    mask_size: size of each mask in time and frequency
    fill_value: value to fill in the masked area
    """
    spec_shape = tf.shape(spec)
    H = spec_shape[0]
    W = spec_shape[1]
    
    for _ in range(num_masks):
        # Randomly choose start positions
        t_start = tf.random.uniform([], minval=0, maxval=W - mask_size, dtype=tf.int32)
        f_start = tf.random.uniform([], minval=0, maxval=H - mask_size, dtype=tf.int32)
        
        # 1. Create the "hole" (block of ones)
        patch = tf.ones([mask_size, mask_size, 2])
        
        # 2. Calculate padding to place the patch at (f_start, t_start)
        # paddings = [[top, bottom], [left, right], [channels_before, channels_after]]
        paddings = [
            [f_start, H - f_start - mask_size], 
            [t_start, W - t_start - mask_size], 
            [0, 0]
        ]
        
        # 3. Create the full-size mask
        mask = tf.pad(patch, paddings)
        # 4. Apply the mask
        spec = tf.where(mask==1, tf.fill(tf.shape(spec), fill_value), spec)
        
    return spec

def random_time_crop(spec, percent=0.3):
    """
    Randomly crops a segment of the spectrogram in the time dimension.
    spec: (H, W, C)
    percent: percentage of width to crop
    
    Returns (H, W, C) where extra widths is filled with 0.0s
    """
    if percent <= 0.0 or percent >= 1.0:
        return spec
    spec_shape = tf.shape(spec)
    H = spec_shape[0]
    W = spec_shape[1]
    
    crop_w = tf.cast(tf.cast(W, tf.float32) * percent, tf.int32)
    
    start = tf.random.uniform([], minval=0, maxval=W - crop_w, dtype=tf.int32)
    cropped = spec[:, start:start+crop_w, :]
    
    # Pad to original width
    pad_left = start
    pad_right = W - (start + crop_w)
    paddings = [[0, 0], [pad_left, pad_right], [0, 0]]
    cropped = tf.pad(cropped, paddings, constant_values=0.0)
    
    return cropped

def random_time_dilation(spec, scale_range=(0.8, 1.2)):
    """
    Randomly time-stretches the spectrogram.
    spec: (H, W, C)
    scale_range: tuple of (min_scale, max_scale)
    
    Returns (H, W, C) resized back to original width.
    """
    spec_shape = tf.shape(spec)
    H = spec_shape[0]
    W = spec_shape[1]
    
    scale = tf.random.uniform([], minval=scale_range[0], maxval=scale_range[1])
    new_W = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)
    
    # Resize to new width
    dilated = tf.image.resize(spec, [H, new_W])
    
    # Reshape back to original width
    if new_W < W:
        # Pad
        pad_right = W - new_W
        paddings = [[0, 0], [0, pad_right], [0, 0]]
        dilated = tf.pad(dilated, paddings, constant_values=0.0)
    else:
        # Crop
        dilated = dilated[:, :W, :] 
    
    return dilated

def random_pitch_shift(spec, max_shift=5):
    """
    Randomly shifts the pitch of the spectrogram.
    spec: (H, W, C)
    max_shift: maximum number of semitones to shift (positive or negative)
    
    Returns (H, W, C)
    """
    shift = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)
    
    # shift the spectrogram along the frequency axis
    spec_shape = tf.shape(spec)
    H = spec_shape[0]
    W = spec_shape[1]
    pad = tf.zeros([shift, W, spec_shape[2]]) if shift > 0 else tf.zeros([-shift, W, spec_shape[2]])
    if shift > 0:
        shifted = tf.concat([spec[:-shift, :, :], pad], axis=0)
    else:
        shifted = tf.concat([pad, spec[-shift:, :, :]], axis=0)
    
    return shifted

def random_drop_channels(spec, prob=0.1):
    """
    Randomly drops one of the channels (Left or Right) with a given probability.
    spec: (H, W, C)
    prob: probability of dropping a channel
    
    Returns (H, W, C)
    """
    drop = tf.random.uniform([], 0, 1) < prob
    if drop:
        channel_to_drop = tf.random.uniform([], 0, 2, dtype=tf.int32)
        if channel_to_drop == 0:
            spec = tf.concat([tf.zeros_like(spec[:, :, 0:1]), spec[:, :, 1:2]], axis=2)
        else:
            spec = tf.concat([spec[:, :, 0:1], tf.zeros_like(spec[:, :, 1:2])], axis=2)
    return spec

def random_conv2d(spec, magnitude=0.5):
    """
    Applies a random 2D convolution to the spectrogram.
    spec: (H, W, C)
    
    Returns (H, W, C)
    """
    spec_expanded = tf.expand_dims(spec, axis=0) # (1, H, W, C)
    C = tf.shape(spec)[2]
    
    # Random kernel size (e.g., 3, 4, 5, 6)
    k_size = tf.random.uniform([], minval=3, maxval=7, dtype=tf.int32)
    
    # Create random filters manually
    # Shape: [filter_height, filter_width, in_channels, out_channels]
    filters = tf.random.normal([k_size, k_size, C, 1], mean=0.0, stddev=0.1)
    
    # Apply convolution
    convolved = tf.nn.conv2d(
        spec_expanded,
        filters,
        strides=[1, 1, 1, 1],
        padding='SAME'
    ) # Output shape: (1, H, W, 1)
    
    convolved = tf.squeeze(convolved, axis=0) # (H, W, 1)
    
    # Add to original spec
    # spec is (H, W, C), convolved is (H, W, 1). Broadcasting handles the addition.
    summed = tf.clip_by_value((1 - magnitude) * spec + magnitude * convolved, 0.0, 1.0)
    return summed

# ==============================================================================
# 5. Dataset Pipeline
# ==============================================================================
def get_dataset(data_dir, batch_size):
    """
    Creates the full training dataset pipeline.
    """
    # 1. List files (Lazy)
    file_pattern = str(data_dir) + "/*.m4a" 
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    
    # 2. Load Audio
    # Use interleave to flatten chunks from each file
    ds = ds.interleave(
        load_audio_dataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # 3. Convert to Spectrogram
    ds = ds.map(make_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Generate Views
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
        # Increased figure width to accommodate stereo views
        fig, axes = plt.subplots(2, (num_views + 1) // 2, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(num_views):
            # Get the i-th view
            # Shape: (H, W, C)
            spec_l = views[i, :, :, 0].numpy()
            spec_r = views[i, :, :, 1].numpy()
            
            # Flip Y axis so low freq is at bottom
            spec_l = np.flipud(spec_l)
            spec_r = np.flipud(spec_r)
            
            # Concatenate Left and Right channels horizontally
            # Add a small separator (10 pixels wide) using the min value (background color)
            separator = np.full((spec_l.shape[0], 10), np.min(spec_l))
            combined = np.concatenate([spec_l, separator, spec_r], axis=1)
            
            ax = axes[i]
            im = ax.imshow(combined, aspect='auto', cmap='magma')
            ax.set_title(f"View {i} {'(Global)' if i < NUM_GLOBAL_VIEWS else '(Local)'} [Left | Right]")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        break

visualize_sample("/Users/elly/Desktop/T7/test")