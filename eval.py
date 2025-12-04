import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt

from models.transformer import ViT
from models.probe import LinearProbe
from data.dataset import get_dataset
from constants import CONFIG, AUDIOSET_NUM_CLASSES
from tqdm import tqdm 

def compute_map_numpy(y_true, y_score):
    """
    Pure NumPy implementation of mAP (macro-averaged across classes).
    y_true, y_score: (N, C)
    """
    eps = 1e-8
    N, C = y_true.shape
    ap_list = []
    for c in range(C):
        y = y_true[:, c]
        s = y_score[:, c]
        if np.sum(y) == 0:
            # Skip classes with no positive examples. Use 0 for a conservative estimate.
            ap_list.append(0.0)
            continue
        order = np.argsort(-s)
        y_sorted = y[order]
        # Accumulate TP/FP
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        recall = tp / (np.sum(y) + eps)
        precision = tp / (tp + fp + eps)
        # Compute interpolated AP via the precision-recall curve (point-wise envelope).
        # Make precision non-increasing (upper envelope).
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        # Sum over recall change points
        idx = np.where(np.diff(recall, prepend=0) > 0)[0]
        ap = np.sum((recall[idx] - recall[idx - 1]) * precision[idx])
        ap_list.append(ap)
    return float(np.mean(ap_list))

def evaluate(data_dir, csv_path, ckpt_dir, batch_size):
    # Strategy: use the default distribution strategy; works on CPU and single/multi-GPU.
    strategy = tf.distribute.get_strategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Dataset (evaluation mode: no shuffle, no multi-view augmentation, V=1)
    ds = get_dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        training=False
    )

    with strategy.scope():
        model = ViT(CONFIG)
        probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)

        # Restore checkpoint (model and probe only; optimizer is not required)
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
        print(f"Restoring from: {latest}")
        tf.train.Checkpoint(model=model, probe=probe).restore(latest).expect_partial()

    y_true_list = []
    y_score_list = []

    @tf.function
    def forward_step(batch_views):
        # batch_views: (B, 1, H, W, C)
        outputs = model(batch_views, training=False)          # (B, 1, Seq, D)
        
        # patch_embeds = outputs[:, 0, 1:, :][0].numpy()       # (N_patches, D)

        # # 4) Patch grid size
        # ph, pw, po = CONFIG.patch_height, CONFIG.patch_width, CONFIG.patch_overlap
        # num_rows = (CONFIG.image_height - po) // (ph - po)
        # num_cols = (CONFIG.image_width  - po) // (pw - po)
        # expected = int(num_rows * num_cols)
        # if patch_embeds.shape[0] != expected:
        #     raise RuntimeError(f"Patch count mismatch: got {patch_embeds.shape[0]}, expected {expected}")

        # # 5) PCA -> 3D -> [0,1] RGB
        # pca = PCA(n_components=3, svd_solver="auto", random_state=0)
        # rgb = pca.fit_transform(patch_embeds)                # (N_patches, 3)
        # rgb = rgb - rgb.mean(axis=0, keepdims=True)
        # rgb_min = rgb.min(axis=0, keepdims=True)
        # rgb_max = rgb.max(axis=0, keepdims=True)
        # rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
        # rgb = np.clip(rgb, 0.0, 1.0)

        # # 6) Reshape to grid and upsample
        # color_grid = rgb.reshape(num_rows, num_cols, 3)      # (rows, cols, 3)
        # color_big = tf.image.resize(
        #     color_grid[None, ...],
        #     [CONFIG.image_height, CONFIG.image_width],
        #     method="nearest"
        # )[0].numpy()                                         # (H, W, 3)

        # # 7) Background spectrogram and overlay
        # spec = batch_views[0, 0].numpy()                     # (H, W, C)
        # spec_gray = spec.mean(axis=-1)                       # (H, W)

        # plt.figure(figsize=(10, 4))

        # # Left: patch-level color grid
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.flipud(color_grid), aspect="auto")
        # plt.title("Patch PCA Coloring (grid)")
        # plt.xlabel("Frequency (patch cols)")
        # plt.ylabel("Time (patch rows)")

        # # Right: overlay on spectrogram
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.flipud(spec_gray), cmap="gray", aspect="auto")
        # plt.imshow(np.flipud(color_big), alpha=0.45, aspect="auto")
        # plt.title("PCA Colors over Spectrogram")
        # plt.axis("off")

        # plt.tight_layout()
        # plt.show()
        
        
        
        
        cls_embed = outputs[:, 0, 0, :]                       # (B, D)
        logits = probe(cls_embed)                              # (B, C)
        probs = tf.nn.sigmoid(logits)                          # (B, C)
        # tf.print(probs[0])
        tf.print(cls_embed[0])
        return probs

    for batch_views, batch_labels in tqdm(ds, desc="Evaluating"):
        # shape: batch_views: (B, 1, H, W, C), batch_labels: (B, 1, CLASSES)
        probs = forward_step(batch_views)
        batch_labels = tf.squeeze(batch_labels, axis=1)  # (B, CLASSES)
        y_true_list.append(batch_labels.numpy())
        y_score_list.append(probs.numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)

    # Compute mAP (prefer sklearn if available)
    mAP = None
    try:
        from sklearn.metrics import average_precision_score
        mAP = float(average_precision_score(y_true, y_score, average="macro"))
    except Exception:
        mAP = compute_map_numpy(y_true, y_score)

    # Optional: multi-label binary accuracy at 0.5 threshold (coarse reference)
    bin_pred = (y_score >= 0.5).astype(np.float32)
    bin_acc = float((bin_pred == y_true).mean())

    print("\n========== Evaluation ==========")
    print(f"Samples: {y_true.shape[0]}, Classes: {y_true.shape[1]}")
    print(f"mAP (macro): {mAP:.4f}")
    print(f"Binary accuracy@0.5: {bin_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="downloads/audioset/eval_segments", help="Directory containing evaluation audio files (.m4a)")
    parser.add_argument("--csv_path", default="data/audioset/eval_segments.csv", help="Path to the AudioSet eval CSV")
    parser.add_argument("--ckpt_dir", default="./checkpoints", help="Directory of saved training checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()