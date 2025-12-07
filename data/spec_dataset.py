import tensorflow as tf
import librosa
import numpy as np
import os
from models.transformer import mask_freq_band, mask_patches, mask_timeframe
from constants import *
import pandas as pd
from tqdm import tqdm

# Unlike dataset.py which loads in m4a or wav files,
# this module focuses on loading preprocessed spectrogram .npy files
# and provides utilities for label loading and spectrogram augmentation.
# ==============================================================================

LABEL_MAP = {} # maps YTID to label vector
CSV_LOADED = set() # to avoid re-loading CSVs multiple times
LABEL_NAMES = [] # maps index to label name

# ==============================================================================
# 1. Setup label lookup table
# ==============================================================================
def load_audioset_label_map(segments_csv_path):
    # every row of data in csv has
    # # YTID, start_seconds, end_seconds, positive_labels
    
    # each label can be found in data/audioset/class_labels_indices.csv
    df = pd.read_csv(segments_csv_path, quotechar='"', skipinitialspace=True, header=2, )
    label_df = pd.read_csv("data/audioset/class_labels_indices.csv", quotechar='"', skipinitialspace=True, )
    global LABEL_NAMES
    LABEL_NAMES = label_df['display_name'].tolist()
    # create a map from YTID to label (binary vector of length 527)
    global LABEL_MAP
    global CSV_LOADED
    if segments_csv_path in CSV_LOADED: # if already loaded, do nothing
        return 
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading Audioset labels from {segments_csv_path}"):
        yt_id = row['# YTID']
        labels = row['positive_labels'].split(',')
        label_vector = np.zeros((AUDIOSET_NUM_CLASSES,), dtype=np.float32)
        
        for label in labels:
            label_index = label_df[label_df['mid'] == label]['index'].values
            if len(label_index) > 0:
                label_vector[label_index[0]] = 1.0
        
        LABEL_MAP[yt_id] = label_vector

    CSV_LOADED.add(segments_csv_path)
    return 



# ==============================================================================
# 2. Spectrogram Loading (Python Function)
# ==============================================================================
def python_load_spectrogram(file_path_tensor):
    """
    Loads a pre-computed spectrogram .npy file.
    """
    file_path = file_path_tensor.numpy().decode('utf-8')
    file_name = str(os.path.basename(file_path))
    
    # Extract YTID from filename: YTID_chunkIndex.npy
    # e.g., "abc123_0.npy" -> "abc123"
    # We split by '_' and take everything except the last part (chunk index)
    # This handles cases where YTID itself might contain underscores (though rare in AudioSet)
    name_no_ext = os.path.splitext(file_name)[0]
    yt_id = name_no_ext.rsplit('_', 1)[0]
    
    # look up label
    label_vec = LABEL_MAP.get(yt_id, np.zeros((AUDIOSET_NUM_CLASSES,), dtype=np.float32))
    
    try:
        # Load numpy file
        # Shape: (H, W, C)
        spec = np.load(file_path)
        spec = spec.astype(np.uint8)
        # Convert to float32 and normalize back to [0, 1]
        # We saved as uint8 [0, 255], so divide by 255.0
        spec = spec.astype(np.float32) / 255.0
        # ensure shape
        H, W, C = np.shape(spec)
        if H != IMAGE_HEIGHT or W != IMAGE_WIDTH:
            spec = tf.image.resize(tf.convert_to_tensor(spec), [IMAGE_HEIGHT, IMAGE_WIDTH], method='bilinear')
        return spec, label_vec
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return dummy zero spectrogram
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), dtype=np.float32), np.zeros((AUDIOSET_NUM_CLASSES,), dtype=np.float32)

def load_spectrogram_dataset(file_path):
    """
    Wraps the python function and returns a Dataset.
    """
    [spec, label] = tf.py_function(
        func=python_load_spectrogram,
        inp=[file_path],
        Tout=[tf.float32, tf.float32],
    )
    # Set shape: (H, W, C)
    spec.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    label.set_shape([AUDIOSET_NUM_CLASSES])
    
    return tf.data.Dataset.from_tensors((spec, label))

# ==============================================================================
# 4. View Generation (Augmentation)
# ==============================================================================
def generate_views(spectrogram, label):
    """
    Takes one full spectrogram and generates V views.
    Input: (H, W, C), (NUM_CLASSES,)
    Output: (V, H, W, C), (V, NUM_CLASSES)
    """
    views = []
    
    # 1. Global Views
    for _ in range(NUM_GLOBAL_VIEWS):
        # mask handled at patch-level masking in transformer
        shifted = random_pitch_shift(spectrogram, max_shift=2)
        dilated = random_time_dilation(shifted, scale_range=(1.0, 1.1))
        scaled = random_volumn_scale(dilated, scale_range=(0.8, 1.2))
        views.append(scaled)
        
    # 2. Local Views
    for _ in range(NUM_LOCAL_VIEWS):
        # time crop handled at patch-level masking in transformer
        dilated = random_time_dilation(spectrogram, scale_range=(0.7, 1.3))
        shifted = random_pitch_shift(spectrogram, max_shift=10)
        conved = random_conv2d(shifted, magnitude=0.2)
        
        # mask handled at patch-level masking in transformer
        noised = random_noise(conved, noise_level=0.03)
        scaled = random_volumn_scale(noised, scale_range=(0.6, 1.4))
        dropped = random_drop_channels(scaled, prob=0.05)
        views.append(dropped)
    
    label = tf.expand_dims(label, axis=0)  # (1, NUM_CLASSES)
    return tf.stack(views), tf.tile(label, [NUM_GLOBAL_VIEWS + NUM_LOCAL_VIEWS, 1]) # (V, NUM_CLASSES)

def random_mask(spec, num_masks=1, mask_size=20, fill_value=0.0):
    """
    Applies image-level random time-frequency masks to the spectrogram.
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
    Randomly apply image-level crops to a segment of the spectrogram in the time dimension.
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

def random_noise(spec, noise_level=0.05):
    """
    Adds random Gaussian noise to the spectrogram.
    spec: (H, W, C)
    noise_level: standard deviation of the Gaussian noise
    
    Returns (H, W, C)
    """
    
    # randomize noise level
    noise_level += tf.random.uniform([], 0.0, noise_level*2)
    noise = tf.random.normal(tf.shape(spec), mean=0.0, stddev=noise_level)
    noised = tf.clip_by_value(spec + noise, 0.0, 1.0)
    return noised

def random_volumn_scale(spec, scale_range=(0.8, 1.2)):
    """
    Randomly scales the volume of the spectrogram.
    spec: (H, W, C)
    scale_range: tuple of (min_scale, max_scale)
    
    Returns (H, W, C)
    """
    scale = tf.random.uniform([], minval=scale_range[0], maxval=scale_range[1])
    scaled = tf.clip_by_value(spec * scale, 0.0, 1.0)
    return scaled
# ==============================================================================
# 5. Dataset Pipeline
# ==============================================================================
def preprocess_batch(spec_batch, label_batch, training=True):
    """
    Applies view generation to a whole batch of spectrograms.
    Designed to run on GPU.
    """
    # Note: Input is already spectrograms (Batch, H, W, C)
    
    if training:
        # Generate Views (Augmentation)
        views, labels = tf.map_fn(
            lambda x: generate_views(x[0], x[1]),
            (spec_batch, label_batch),
            fn_output_signature=(tf.float32, tf.float32),
            parallel_iterations=32
        )
    else:
        # Expand dims for eval
        def expand_view(spec, label):
            spec = tf.expand_dims(spec, axis=0) # (1, H, W, C)
            label = tf.expand_dims(label, axis=0) # (1, NUM_CLASSES)
            return spec, label
            
        views, labels = tf.map_fn(
            lambda x: expand_view(x[0], x[1]),
            (spec_batch, label_batch),
            fn_output_signature=(tf.float32, tf.float32),
            parallel_iterations=32
        )

    return views, labels

def get_dataset(data_dir, csv_path, batch_size, dataset="audioset", training=True):
    """
    Creates the full training dataset pipeline with GPU acceleration.
    """
    global LABEL_MAP
    global CSV_LOADED
    
    if dataset == "audioset":
        load_audioset_label_map(csv_path)
        # 1. List files (Lazy)
        # Match .npy files
        file_pattern = str(data_dir) + "/*.npy" 
        ds = tf.data.Dataset.list_files(file_pattern, shuffle=training)
        
        # 2. Load Spectrograms
        # Use interleave to load files in parallel
        ds = ds.interleave(
            load_spectrogram_dataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=(not training)
        )
        
        # 3. Batch FIRST (Critical for GPU efficiency)
        ds = ds.batch(batch_size, drop_remainder=True)
        
        # 4. Move to GPU and Compute Augmentations
        gpu_map_func = lambda x, y: preprocess_batch(x, y, training=training)
        
        # Explicitly place on GPU
        # with tf.device("/device:GPU:0"):
        ds = ds.map(gpu_map_func, num_parallel_calls=tf.data.AUTOTUNE)
        
        # 5. Prefetch
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def visualize_sample(data_dir, csv_path):
    """
    Takes one sample from the dataset and plots the generated spectrograms (views).
    """
    import matplotlib.pyplot as plt
    
    # Create a small dataset with batch_size=1
    ds = get_dataset(data_dir, csv_path, batch_size=1)
    
    # Take 1 batch
    for batch, label in ds.take(1):
        # batch shape: (1, TOTAL_VIEWS, H, W, C)
        # Preview patch masking 
        # 1. Split into patches
        batch = tf.reshape(batch, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]) # (B*V, H, W, C)

        patches = tf.image.extract_patches(
            images=batch,
            sizes=[1, CONFIG.patch_height, CONFIG.patch_width, 1],
            strides=[1, CONFIG.patch_height - CONFIG.patch_overlap, CONFIG.patch_width - CONFIG.patch_overlap, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        batch_size = tf.shape(patches)[0]
        num_row_patches = tf.shape(patches)[1]
        num_col_patches = tf.shape(patches)[2]
        # patches now has shape (batch_size * V, num_rows_patches, num_cols_patches, patch_height * patch_width * num_channels)
        x = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1] * tf.shape(patches)[2], tf.shape(patches)[3]])  # flatten patches
        # 2. Apply masking
        mask = mask_patches(batch_size, num_row_patches, num_col_patches, CONFIG.G, CONFIG.V)
        mask |= mask_timeframe(batch_size, num_row_patches, num_col_patches, CONFIG.G, CONFIG.V)
        mask |= mask_freq_band(batch_size, num_row_patches, num_col_patches, CONFIG.G, CONFIG.V)
        x = tf.where(mask, tf.zeros_like(x), x)
        # 3. Unflatten patches
        patches = tf.reshape(x, [batch_size, num_row_patches, num_col_patches, CONFIG.patch_height, CONFIG.patch_width, CONFIG.num_channels])
        # 4. Reconstruct images
        # batch_size here is actually B * V because of the earlier reshape
        out = tf.zeros((batch_size, CONFIG.image_height, CONFIG.image_width, CONFIG.num_channels))
        # add patches one by one
        for i in range(num_row_patches):
            for j in range(num_col_patches):
                patch = patches[:, i, j, :, :, :] # shape: (batch_size, patch_height, patch_width, num_channels)
                if i < num_row_patches - 1 and CONFIG.patch_overlap > 0:
                    patch = patch[:, :-CONFIG.patch_overlap, :, :]
                if j < num_col_patches - 1 and CONFIG.patch_overlap > 0:
                    patch = patch[:, :, :-CONFIG.patch_overlap, :]
                top_left_corner_icoor = (CONFIG.patch_height - CONFIG.patch_overlap) * i
                top_left_corner_jcoor = (CONFIG.patch_width - CONFIG.patch_overlap) * j
                
                # Pad for (Batch, Height, Width, Channels)
                # Use tf.stack to ensure we create a valid tensor from mixed ints/tensors
                pad_h = tf.stack([top_left_corner_icoor, CONFIG.image_height - (top_left_corner_icoor + tf.shape(patch)[1])])
                pad_w = tf.stack([top_left_corner_jcoor, CONFIG.image_width - (top_left_corner_jcoor + tf.shape(patch)[2])])
                pad = tf.stack([
                    tf.constant([0, 0], dtype=tf.int32),
                    pad_h,
                    pad_w,
                    tf.constant([0, 0], dtype=tf.int32)
                ])
                # tf.print("Patching at:", top_left_corner_icoor, top_left_corner_jcoor, "with pad:", pad)
                padded_patch = tf.pad(patch, pad)
                out += padded_patch
        out = tf.reshape(out, [-1, CONFIG.V, CONFIG.image_height, CONFIG.image_width, CONFIG.num_channels])
        batch = out     
        
        # Take exactly one sample
        views = batch[0] # (TOTAL_VIEWS, H, W, C)
        label_vec = label[0] # (TOTAL_VIEWS, AUDIOSET_NUM_CLASSES)
        label_vec = label_vec[0] # (AUDIOSET_NUM_CLASSES,)
        label_names = [LABEL_NAMES[i] for i in range(AUDIOSET_NUM_CLASSES) if label_vec[i] == 1.0]
        num_views = views.shape[0]
        
        # Plot
        # Increased figure width to accommodate stereo views
        fig, axes = plt.subplots((num_views + 1) // 2, 2, figsize=(8, 8/CONFIG.image_width*CONFIG.image_height))
        fig.suptitle(f"Generated Views (Labels: {', '.join(label_names)})", fontsize=12)
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
            im = ax.imshow(combined, aspect='auto', cmap='viridis')
            ax.set_title(f"View {i} {'(Global)' if i < NUM_GLOBAL_VIEWS else '(Local)'} [Left | Right]")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        break

# visualize_sample(DATA_DIR, CSV_PATH)
