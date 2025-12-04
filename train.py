import os
import time
import wandb
from dataclasses import asdict
import tensorflow as tf
from models.transformer import ViT, ViTConfig
from models.loss import LeJEPA
from models.probe import LinearProbe
from data.dataset import get_dataset
from constants import *
from tqdm import tqdm

try:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    print("Mixed precision (bfloat16) enabled.")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

# ==============================================================================
# 2. Setup Distributed Strategy
# ==============================================================================
# MirroredStrategy handles both single-GPU and multi-GPU scenarios.
# But on Mac's Metal backend, MirroredStrategy triggers a known bug. 
# So use default strategy for debugging on Mac.

# If on CUDA, use MirroredStrategy
if len(tf.config.list_physical_devices('GPU')) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print("Using MirroredStrategy for multi-GPU training.")
else:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (single device or non-GPU).")
print(f"Number of devices: {strategy.num_replicas_in_sync}")


# Create and distribute the dataset
# experimental_distribute_dataset splits the global batch among replicas
with strategy.scope():
    dataset = get_dataset(
        data_dir=DATA_DIR, 
        csv_path=CSV_PATH,
        batch_size=GLOBAL_BATCH_SIZE,
        training=True
    )
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    tot_num_batches = (17895 * 2 + GLOBAL_BATCH_SIZE - 1) // GLOBAL_BATCH_SIZE
# ============================================================================== 
# 4. Model, Optimizer, and Loss Initialization
# ============================================================================== 
with strategy.scope():
    # Initialize Model
    model = ViT(CONFIG)
    probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)
    # Initialize Optimizer
    steps_per_epoch = tot_num_batches
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = steps_per_epoch * 5 # warmup
    decay_steps = total_steps - warmup_steps
    initial_learning_rate = 0.0
    target_learning_rate = BASE_LEARNING_RATE
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_warmup_decayed_fn, weight_decay=1e-2)
    
    # Initialize Loss
    loss_fn = LeJEPA(G=NUM_GLOBAL_VIEWS, V=TOTAL_VIEWS, lambd=LAMBDA_SIGREG)
    
    # Checkpoint Manager (Optional but recommended)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, probe=probe)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=5
    )
    
    # print model summary
    dummy_input = tf.zeros((1, 1, CONFIG.image_height, CONFIG.image_width, CONFIG.num_channels))
    _ = model(dummy_input, training=False)
    _ = probe(tf.zeros((1, CONFIG.hidden_dim)), training=False)
    model.summary()
    probe.summary()

def balanced_acc(y_true, y_pred):
    """
    Computes balanced accuracy for multi-label classification.
    y_true, y_pred: Tensors of shape (Batch, Num_Classes)
    """
    y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)
    
    true_positives = tf.reduce_sum(y_true * y_pred_binary, axis=0)
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary), axis=0)
    
    positives = tf.reduce_sum(y_true, axis=0)
    negatives = tf.reduce_sum(1 - y_true, axis=0)
    
    sensitivity = tf.where(positives > 0, true_positives / positives, 0.0)
    specificity = tf.where(negatives > 0, true_negatives / negatives, 0.0)
    
    num_valid_positive_classes = tf.reduce_sum(tf.cast(positives > 0, tf.float32))
    num_valid_negative_classes = tf.reduce_sum(tf.cast(negatives > 0, tf.float32))
    
    avg_sensitivity = tf.reduce_sum(sensitivity) / (num_valid_positive_classes + 1e-8)
    avg_specificity = tf.reduce_sum(specificity) / (num_valid_negative_classes + 1e-8)
    
    balanced_accuracy = (avg_sensitivity + avg_specificity) / 2.0
    
    return balanced_accuracy

# ============================================================================== 
# 5. Training Step
# ============================================================================== 
def train_step(inputs):
    """
    Runs on EACH replica.
    inputs: Tuple of (views, labels)
        views: (Per_Replica_Batch_Size, TOTAL_VIEWS, H, W, C)
        labels: (Per_Replica_Batch_Size, TOTAL_VIEWS, NUM_CLASSES)
    """
    views, labels = inputs
    
    # 1. Flatten views into the batch dimension for the model
    # Shape: (B * V, H, W, C)
    B = tf.shape(views)[0]
    V = TOTAL_VIEWS
    
    with tf.GradientTape(persistent=True) as tape:
        # 2. Forward Pass
        # Input: (B, V, H, W, C)
        # Output: (B, V, Seq_Len, Hidden_Dim)
        outputs = model(views, training=True)
        
        # 3. Extract CLS Tokens
        cls_tokens = outputs[:, :, 0, :]
        
        # 4. Prepare inputs for LeJEPA Loss
        all_emb = tf.reshape(cls_tokens, [B * V, -1])
        
        # global_emb: Needs to be (B * G, D)
        # Slice the first G views (assuming they are the global ones)
        global_views = cls_tokens[:, :NUM_GLOBAL_VIEWS, :] # (B, G, D)
        
        # Flatten back to (B * G, D)
        global_emb = tf.reshape(global_views, [B * NUM_GLOBAL_VIEWS, -1])
        
        # 5. Calculate LeJEPA Loss (Backbone)
        # LeJEPA returns a scalar (sum of inv_loss and sigreg_loss)
        # We call .call() directly to pass the step argument, bypassing keras.Loss.__call__
        per_replica_loss_backbone = loss_fn.call(global_emb, all_emb, step=optimizer.iterations)
        
        # 6. Calculate Probe Loss (Linear Probe)
        # Input to probe: Average of Global Views
        # We use stop_gradient to ensure backbone is FROZEN for this part
        probe_input = tf.stop_gradient(tf.reduce_mean(global_views, axis=1)) # (B, D)
        
        probe_logits = probe(probe_input, training=True) # (B, Num_Classes)
        
        # Labels: Take the label of the first view (since all views have same label)
        batch_labels = labels[:, 0, :] # (B, Num_Classes)
        
        # Binary Cross Entropy for Multi-label classification
        # Sum over classes, mean over batch
        # Cast probe_logits to float32 for loss calculation stability
        probe_logits = tf.cast(probe_logits, tf.float32)
        
        per_replica_loss_probe = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.weighted_cross_entropy_with_logits(labels=batch_labels, logits=probe_logits, pos_weight=20.0),
                axis=1
            )
        )
        
        # 7. Scale Losses for Global Batch
        scaled_loss_backbone = per_replica_loss_backbone / float(strategy.num_replicas_in_sync)
        scaled_loss_probe = per_replica_loss_probe / float(strategy.num_replicas_in_sync)
        
        # 8. Calculate Accuracy (Binary Accuracy for Multi-label)
        preds = tf.nn.sigmoid(probe_logits)
        acc = balanced_acc(batch_labels, preds)
        per_replica_acc = tf.reduce_mean(acc)
        scaled_acc = per_replica_acc / float(strategy.num_replicas_in_sync)

    # 9. Compute and Apply Gradients
    grads_backbone = tape.gradient(scaled_loss_backbone, model.trainable_variables)
    grads_probe = tape.gradient(scaled_loss_probe, probe.trainable_variables)
    
    all_grads = grads_backbone + grads_probe
    all_vars = model.trainable_variables + probe.trainable_variables
    optimizer.apply_gradients(zip(all_grads, all_vars))
    
    del tape # Explicitly delete persistent tape
    
    return scaled_loss_backbone, scaled_loss_probe, scaled_acc

# ============================================================================== 
# 6. Distributed Training Loop
# ============================================================================== 
@tf.function
def distributed_train_step(dataset_inputs):
    # Run train_step on all replicas
    per_replica_loss_backbone, per_replica_loss_probe, per_replica_acc = strategy.run(train_step, args=(dataset_inputs,))
    
    # Aggregate losses (SUM) to get the global average loss
    global_loss_backbone = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_backbone, axis=None)
    global_loss_probe = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_probe, axis=None)
    global_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_acc, axis=None)
    
    return global_loss_backbone, global_loss_probe, global_acc

def main():
    print("Starting training...")
    wandb.init(project="MyGO", config=asdict(CONFIG))
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        start_time = time.time()
        
        total_loss_backbone = 0.0
        total_loss_probe = 0.0
        total_acc = 0.0
        num_batches = 0
        
        # Iterate over the distributed dataset
        for step, batch_inputs in tqdm(enumerate(dist_dataset), total=tot_num_batches, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"):
            loss_backbone, loss_probe, acc = distributed_train_step(batch_inputs)
            
            total_loss_backbone += loss_backbone
            total_loss_probe += loss_probe
            total_acc += acc
            num_batches += 1
            
            if step % LOG_EVERY_STEPS == 0:
                print(f"  Step {step}: Backbone Loss = {loss_backbone:.4f}, Probe Loss = {loss_probe:.4f}, Acc = {acc:.4f}")
                wandb.log({
                    "backbone_loss": loss_backbone,
                    "probe_loss": loss_probe,
                    "probe_accuracy": acc,
                    "epoch": epoch
                })
        
        avg_loss_backbone = total_loss_backbone / num_batches if num_batches > 0 else 0.0
        avg_loss_probe = total_loss_probe / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s.")
        print(f"  Avg Backbone Loss: {avg_loss_backbone:.4f}")
        print(f"  Avg Probe Loss:    {avg_loss_probe:.4f}")
        print(f"  Avg Probe Accuracy: {avg_acc:.4f}")
        
        wandb.log({
            "epoch_avg_backbone_loss": avg_loss_backbone,
            "epoch_avg_probe_loss": avg_loss_probe,
            "epoch_avg_probe_accuracy": avg_acc,
            "epoch": epoch
        })
        
        # Save checkpoint
        checkpoint_manager.save()

if __name__ == "__main__":
    main()
