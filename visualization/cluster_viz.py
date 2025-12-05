import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.transformer import ViT
from models.probe import LinearProbe
from data.dataset import get_dataset
from constants import CONFIG, AUDIOSET_NUM_CLASSES


def cluster_visualize(data_dir, csv_path, ckpt_dir, k=6, sample_idx=0):
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install via: pip install scikit-learn") from e

    # Compute patch grid size consistent with model config
    ph, pw, po = CONFIG.patch_height, CONFIG.patch_width, CONFIG.patch_overlap
    num_rows = (CONFIG.image_height - po) // (ph - po)  # time axis (rows)
    num_cols = (CONFIG.image_width - po) // (pw - po)   # frequency axis (cols)

    # Dataset: evaluation mode gives a single view (V=1)
    ds = get_dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=1,
        training=False
    )

    # Build model and restore weights (probe not used for clustering but included for checkpoint compatibility)
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = ViT(CONFIG)
        probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            print(f"Warning: No checkpoint found in {ckpt_dir}. Using randomly initialized model.")
        else:
            print(f"Restoring from: {latest}")
            tf.train.Checkpoint(model=model, probe=probe).restore(latest).expect_partial()
                
    # Show more examples
    # Take the requested sample (iterate sample_idx times)
    it = iter(ds)
    for _ in range(sample_idx + 1):
        batch_views, _labels = next(it)  # (1, 1, H, W, C)

        # Forward pass to get patch embeddings: outputs -> (B, 1, Seq, D)
        outputs = model(batch_views, training=False)
        patch_embeds = outputs[:, 0, 1:, :]  # remove CLS -> (B, Seq-1, D)
        patch_embeds = patch_embeds[0].numpy()  # (Seq-1, D)

        # Validate patch count
        expected_patches = int(num_rows * num_cols)
        if patch_embeds.shape[0] != expected_patches:
            raise RuntimeError(
                f"Patch count mismatch: got {patch_embeds.shape[0]}, expected {expected_patches} "
                f"(rows={num_rows}, cols={num_cols})."
            )

        # KMeans clustering over patch embeddings
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(patch_embeds).astype(np.int32)  # (Seq-1,)

        # Reshape labels to (rows, cols) grid
        cluster_map = labels.reshape(num_rows, num_cols)

        # Prepare spectrogram for overlay (grayscale by channel mean)
        spec = batch_views[0, 0].numpy()  # (H, W, C)
        spec_gray = spec.mean(axis=-1)    # (H, W)

        # Resize cluster map to input resolution using nearest-neighbor
        cluster_big = tf.image.resize(
            cluster_map[None, :, :, None].astype(np.float32),
            [CONFIG.image_height, CONFIG.image_width],
            method='nearest'
        )[0, :, :, 0].numpy()

        # Plot: left = cluster grid, right = overlay on spectrogram
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cluster_map, cmap='tab20', origin='lower', aspect='auto')
        plt.title(f'Patch KMeans (k={k})')
        plt.xlabel('Frequency (patch cols)')
        plt.ylabel('Time (patch rows)')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 3, 2)
        # Flip vertically so low frequencies are at the bottom
        plt.imshow(np.flipud(spec_gray), cmap='gray', aspect='auto')
        overlay = np.flipud(cluster_big)
        plt.imshow(overlay, cmap='tab20', alpha=0.35, aspect='auto')
        plt.title('Clusters over Spectrogram')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # show original spectrogram
        plt.imshow(np.flipud(spec_gray), cmap='gray', aspect='auto')
        plt.title('Original Spectrogram')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Cluster visualization (KMeans) on ViT patch embeddings.")
    parser.add_argument("--data_dir", default="downloads/audioset/eval_segments", help="Directory of .m4a files")
    parser.add_argument("--csv_path", default="data/audioset/eval_segments.csv", help="AudioSet eval CSV path")
    parser.add_argument("--ckpt_dir", default="./checkpoints", help="Directory of saved checkpoints")
    parser.add_argument("--clusters", type=int, default=6, help="Number of KMeans clusters (k)")
    parser.add_argument("--sample_idx", type=int, default=4, help="Dataset sample index to visualize")
    args = parser.parse_args()

    cluster_visualize(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        ckpt_dir=args.ckpt_dir,
        k=args.clusters,
        sample_idx=args.sample_idx
    )


if __name__ == "__main__":
    main()


