import argparse
import os
import numpy as np
import tensorflow as tf

from models.transformer import ViT
from models.probe import LinearProbe
from data.dataset import get_dataset
from constants import CONFIG, AUDIOSET_NUM_CLASSES

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
        cls_embed = outputs[:, 0, 0, :]                       # (B, D)
        logits = probe(cls_embed)                              # (B, C)
        probs = tf.nn.sigmoid(logits)                          # (B, C)
        return probs

    for batch_views, batch_labels in ds:
        probs = forward_step(batch_views)
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