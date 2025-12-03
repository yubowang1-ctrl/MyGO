from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_height: int
    image_width: int
    num_channels: int
    patch_height: int
    patch_width: int
    patch_overlap: int
    num_layers: int
    hidden_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float
    G: int = 2  # default number of global views
    V: int = 8 # default total number of views
    
# ==============================================================================
# Configuration & Hyperparameters
# ==============================================================================
# Audio Config
TARGET_SR = 48000
DURATION = 10.0 # seconds
SAMPLE_SAMPLES = int(TARGET_SR * DURATION) # 480,000 samples

# Spectrogram Config
N_FFT = 4096 # the frequency resolution is TARGET_SR / N_FFT = ~11.7 Hz
HOP_LEN = 2020 # this number to ensure compatibility with image width and patch sizes     # use int(0.1 * TARGET_SR) if want 0.1 seconds hop
N_MELS = 256
FMIN = 60.0
FMAX = 12000.0
EPSILON = 1e-6

# View Generation Config
NUM_GLOBAL_VIEWS = 2
NUM_LOCAL_VIEWS = 6
TOTAL_VIEWS = NUM_GLOBAL_VIEWS + NUM_LOCAL_VIEWS

# Distributed Training Config
GLOBAL_BATCH_SIZE = 64  # Total batch size across all GPUs
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
LOG_EVERY_STEPS = 10

# Data Config
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 236
NUM_CHANNELS = 2

# LeJEPA Config
NUM_GLOBAL_VIEWS = 2    # G
NUM_LOCAL_VIEWS = 6     # Local views
TOTAL_VIEWS = NUM_GLOBAL_VIEWS + NUM_LOCAL_VIEWS # V
LAMBDA_SIGREG = 0.5

    
# Tiny ViT with 16x16 patches, ~5M parameters
ViT_Ti_16 = ViTConfig(
    image_height=224,
    image_width=224,
    num_channels=3,
    patch_height=16,
    patch_width=16,
    patch_overlap=0,
    num_layers=12,
    hidden_dim=192,
    mlp_dim=768,
    num_heads=3,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
) 

# Small ViT with 16x16 patches, ~22M parameters
ViT_S_16 = ViTConfig(
    image_height=224,
    image_width=224,
    num_channels=3,
    patch_height=16,
    patch_width=16,
    patch_overlap=0,
    num_layers=12,
    hidden_dim=384,
    mlp_dim=1536,
    num_heads=6,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
) 

# Base ViT with 16x16 patches, ~86M parameters
ViT_B_16 = ViTConfig(
    image_height=224,
    image_width=224,
    num_channels=3,
    patch_height=16,
    patch_width=16,
    patch_overlap=0,
    num_layers=12,
    hidden_dim=768,
    mlp_dim=3072,
    num_heads=12,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
)


# Model Config (ViT-Ti-16 as example) 
# this is the config used in train.py!
CONFIG = ViTConfig(
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    num_channels=NUM_CHANNELS,
    patch_height=16,
    patch_width=16,
    patch_overlap=6,
    num_layers=12,
    hidden_dim=192,
    mlp_dim=768,
    num_heads=3,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    G=NUM_GLOBAL_VIEWS,
    V=TOTAL_VIEWS,
)