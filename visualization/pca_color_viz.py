# pca_color_visualize.py
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.transformer import ViT
from models.probe import LinearProbe
from data.dataset import get_dataset
from constants import CONFIG, AUDIOSET_NUM_CLASSES

def pca_color_visualize(data_dir, csv_path, ckpt_dir, sample_idx=0, alpha=0.45):
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install via: pip install scikit-learn") from e

    # 1) Dataset (eval mode -> V=1)
    ds = get_dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=1,
        training=False
    )
    it = iter(ds)
    for _ in range(sample_idx + 1):
        batch_views, _ = next(it)  # (1, 1, H, W, C)

    # 2) Model + checkpoint
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = ViT(CONFIG)
        probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is not None:
            print(f"Restoring from: {latest}")
            tf.train.Checkpoint(model=model, probe=probe).restore(latest).expect_partial()
        else:
            print(f"Warning: No checkpoint found in {ckpt_dir}. Using randomly initialized model.")

    # 3) Forward -> patch embeddings (drop CLS)
    outputs = model(batch_views, training=False)         # (1, 1, Seq, D)
    patch_embeds = outputs[:, 0, 1:, :][0].numpy()       # (N_patches, D)

    # 4) Patch grid size
    ph, pw, po = CONFIG.patch_height, CONFIG.patch_width, CONFIG.patch_overlap
    num_rows = (CONFIG.image_height - po) // (ph - po)
    num_cols = (CONFIG.image_width  - po) // (pw - po)
    expected = int(num_rows * num_cols)
    if patch_embeds.shape[0] != expected:
        raise RuntimeError(f"Patch count mismatch: got {patch_embeds.shape[0]}, expected {expected}")

    # 5) PCA -> 3D -> [0,1] RGB
    pca = PCA(n_components=3, svd_solver="auto", random_state=0)
    rgb = pca.fit_transform(patch_embeds)                # (N_patches, 3)
    rgb = rgb - rgb.mean(axis=0, keepdims=True)
    rgb_min = rgb.min(axis=0, keepdims=True)
    rgb_max = rgb.max(axis=0, keepdims=True)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    rgb = np.clip(rgb, 0.0, 1.0)

    # 6) Reshape to grid and upsample
    color_grid = rgb.reshape(num_rows, num_cols, 3)      # (rows, cols, 3)
    color_big = tf.image.resize(
        color_grid[None, ...],
        [CONFIG.image_height, CONFIG.image_width],
        method="nearest"
    )[0].numpy()                                         # (H, W, 3)

    # 7) Background spectrogram and overlay
    spec = batch_views[0, 0].numpy()                     # (H, W, C)
    spec_gray = spec.mean(axis=-1)                       # (H, W)

    plt.figure(figsize=(10, 4))

    # Left: patch-level color grid
    plt.subplot(1, 2, 1)
    plt.imshow(np.flipud(color_grid), origin="lower", aspect="auto")
    plt.title("Patch PCA Coloring (grid)")
    plt.xlabel("Frequency (patch cols)")
    plt.ylabel("Time (patch rows)")

    # Right: overlay on spectrogram
    plt.subplot(1, 2, 2)
    plt.imshow(np.flipud(spec_gray), cmap="gray", aspect="auto")
    plt.imshow(np.flipud(color_big), alpha=alpha, aspect="auto")
    plt.title("PCA Colors over Spectrogram")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Patch-level PCA coloring over ViT patch embeddings.")
    parser.add_argument("--data_dir", default="downloads/audioset/eval_segments", help="Directory of .m4a files")
    parser.add_argument("--csv_path", default="data/audioset/eval_segments.csv", help="AudioSet eval CSV path")
    parser.add_argument("--ckpt_dir", default="./checkpoints", help="Directory of saved checkpoints")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index to visualize")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay transparency [0,1]")
    args = parser.parse_args()

    pca_color_visualize(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        ckpt_dir=args.ckpt_dir,
        sample_idx=args.sample_idx,
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()