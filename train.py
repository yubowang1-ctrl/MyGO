import os
import sys
import time
import wandb
from dataclasses import asdict
import matplotlib
matplotlib.use("Agg")  # add this line before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.transformer import ViT, ViT_Ti, ViTConfig
from models.loss import LeJEPA
from models.probe import LinearProbe
from data.spec_dataset import get_dataset
from constants import *
from tqdm import tqdm

# silent librosa warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# OSCAR might have issues with bf16
# tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ==============================================================================
# 2. Setup Distributed Strategy
# ==============================================================================
# MirroredStrategy handles both single-GPU and multi-GPU scenarios.
# But on Mac's Metal backend, MirroredStrategy triggers a known bug. 
# So use default strategy for debugging on Mac.

print(tf.config.list_physical_devices('GPU'))

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
    # model = ViT(CONFIG)
    model = ViT_Ti(CONFIG)
    probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)
    # Initialize Optimizer
    steps_per_epoch = tot_num_batches
    total_steps = steps_per_epoch * (NUM_EPOCHS * 2)  # *2 so larger lr for later epochs
    warmup_steps = steps_per_epoch * 5 # warmup
    decay_steps = total_steps - warmup_steps
    initial_learning_rate = 0.0
    target_learning_rate = BASE_LEARNING_RATE
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_warmup_decayed_fn, 
        weight_decay=5e-6,
        beta_1=0.9,
        beta_2=0.999,
        global_clipnorm=0.5 # gradient clipping
    )
    
    # Initialize Loss
    loss_fn = LeJEPA(G=NUM_GLOBAL_VIEWS, V=TOTAL_VIEWS, lambd=LAMBDA_SIGREG)
    
    # --- ADDED: Variable to track best loss ---
    best_loss_var = tf.Variable(float('inf'), trainable=False, dtype=tf.float32, name='best_loss')

    # Checkpoint Manager (Optional but recommended)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, probe=probe, best_loss=best_loss_var)
    
    # 1. Regular Manager (For Resuming - Saves Every Epoch)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=5
    )
    
    # 2. Best Manager (For Inference - Saves Only Best)
    best_checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints/best", max_to_keep=1
    )
    
    latest_ckpt = checkpoint_manager.latest_checkpoint
    start_epoch = 0
    if latest_ckpt:
        print(f"Found checkpoint: {latest_ckpt}")
        
        # 1. Restore EVERYTHING (Model, Probe, Optimizer)
        # This loads the weights and the optimizer's internal state (momentum, step count, etc.)
        status = checkpoint.restore(latest_ckpt)
        
        # Optional: Assert that the restore was successful
        # status.assert_consumed() 
        print("Full checkpoint restored (Backbone, Probe, Optimizer).")
        
        # 2. Parse the epoch number from the filename
        try:
            # Format is usually ".../ckpt-14" -> start_epoch = 14
            ckpt_num = int(latest_ckpt.split('-')[-1])
            start_epoch = ckpt_num
            print(f"Resuming from Epoch {start_epoch + 1}")
        except ValueError:
            print("Could not parse epoch from filename. Starting loop from 0 (but weights are restored).")
            start_epoch = 0
    else:
        print("No checkpoint found. Starting from scratch.")
    
    # print model summary
    dummy_input = tf.zeros((1, 1, CONFIG.image_height, CONFIG.image_width, CONFIG.num_channels))
    _ = model(dummy_input, training=False)
    _ = probe(tf.zeros((1, CONFIG.hidden_dim)), training=False)
    model.summary()
    probe.summary()

    # Explicitly build the optimizer to create variables (slots) 
    # BEFORE the first distributed_train_step.
    # This prevents graph errors in MirroredStrategy during lazy initialization.
    print("Building optimizer variables...")
    all_trainable_vars = model.trainable_variables + probe.trainable_variables
    optimizer.build(all_trainable_vars)
    print("Optimizer built.")

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
        # cls_tokens = bn_for_loss(cls_tokens, training=True)
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
        per_replica_loss_backbone, per_replica_sigreg_loss_backbone = loss_fn.call(global_emb, all_emb, step=optimizer.iterations)
        
        # ============= DEBUGGING CODE FOR SIGREG =============
        # tf.print("std:", tf.math.reduce_std(global_emb[:, 0]), ", ", tf.math.reduce_std(global_emb[:, 1]), "SigReg:", per_replica_sigreg_loss_backbone, output_stream=sys.stdout)
        
        
        # plot the 1 and 2 dimensions of global_emb
        ctx = tf.distribute.get_replica_context()
        cls_2d = global_emb[:, :2]  # (local_B*G, 2)

        # # gather all replicas
        # cls_2d_all = ctx.all_gather(cls_2d, axis=0)

        # def _plot(arr, sigreg_loss):
        #     import numpy as np, matplotlib.pyplot as plt
        #     arr = np.asarray(arr, dtype=np.float32)
        #     plt.figure(figsize=(6,6))
        #     plt.scatter(arr[:,0], arr[:,1], alpha=0.5)
        #     plt.title("CLS dist - std ({:.2f}, {:.2f}) - SIGReg {:.4f}".format(float(np.std(arr[:,0])), float(np.std(arr[:,1])), float(sigreg_loss)) )
        #     plt.savefig(f"figures/cls_token_distribution_step{int(optimizer.iterations)}.png")
        #     plt.close()
        #     return np.int64(0)

        # # run plotting only on one replica to avoid duplicate files
        # def host_plot(v, sigreg_loss):
        #     if ctx.replica_id_in_sync_group == 0:
        #         tf.py_function(_plot, [v, sigreg_loss], Tout=tf.int64)
        #     return v

        # cls_2d_all = host_plot(cls_2d_all, per_replica_sigreg_loss_backbone)
        
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
                tf.nn.weighted_cross_entropy_with_logits(labels=batch_labels, logits=probe_logits, pos_weight=30.0),
                axis=1
            )
        )
        
        # 7. Scale Losses for Global Batch
        scaled_loss_backbone = per_replica_loss_backbone / float(strategy.num_replicas_in_sync)
        scaled_sigreg_loss_backbone = per_replica_sigreg_loss_backbone / float(strategy.num_replicas_in_sync)
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
    
    return scaled_loss_backbone, scaled_loss_probe, scaled_acc, scaled_sigreg_loss_backbone

# ============================================================================== 
# 6. Distributed Training Loop
# ============================================================================== 
@tf.function
def distributed_train_step(dataset_inputs):
    # Run train_step on all replicas
    per_replica_loss_backbone, per_replica_loss_probe, per_replica_acc, per_replica_sigreg_loss_backbone = strategy.run(train_step, args=(dataset_inputs,))
    
    # Aggregate losses (SUM) to get the global average loss
    global_loss_backbone = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_backbone, axis=None)
    global_loss_probe = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_probe, axis=None)
    global_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_acc, axis=None)
    global_sigreg_loss_backbone = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_sigreg_loss_backbone, axis=None)
    return global_loss_backbone, global_loss_probe, global_acc, global_sigreg_loss_backbone

def main():
    print("Starting training...")
    wandb.init(project="MyGO", config=asdict(CONFIG))
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        start_time = time.time()
        
        total_loss_backbone = 0.0
        total_loss_probe = 0.0
        total_acc = 0.0
        total_sigreg_loss_backbone = 0.0
        num_batches = 0
        
        # Iterate over the distributed dataset
        for step, batch_inputs in tqdm(enumerate(dist_dataset), total=tot_num_batches, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"):
            loss_backbone, loss_probe, acc, sigreg_loss_backbone = distributed_train_step(batch_inputs)
            
            total_loss_backbone += loss_backbone
            total_loss_probe += loss_probe
            total_acc += acc
            total_sigreg_loss_backbone += sigreg_loss_backbone
            num_batches += 1
            
            if step % LOG_EVERY_STEPS == 0:
                print(f"  Step {step}: Backbone Loss = {loss_backbone:.4f}, SIGReg Loss = {sigreg_loss_backbone:.4f} Probe Loss = {loss_probe:.4f}, Acc = {acc:.4f}")
                wandb.log({
                    "backbone_loss": loss_backbone,
                    "probe_loss": loss_probe,
                    "probe_accuracy": acc,
                    "epoch": epoch
                })

                try:
                    # --- MODIFIED VISUALIZATION BLOCK ---
                    # Instead of defining a function and using strategy.run, 
                    # we extract the first replica's data and run locally.
                    
                    # 1. Unwrap the distributed batch to get local tensors
                    # batch_inputs is (views, labels)
                    views_replicas = strategy.experimental_local_results(batch_inputs[0])
                    
                    # 2. Take the first replica's data
                    # Shape: (Local_Batch_Size, V, H, W, C)
                    local_views = views_replicas[0]
                    
                    # 3. Run model inference (Training=False)
                    # We use the model directly. Since variables are distributed, 
                    # this works fine for inference on a single batch.
                    outputs = model(local_views, training=False)
                    
                    # 4. Process outputs for plotting
                    cls_tokens = outputs[:, :, 0, :] # (B, V, D)
                    
                    # Flatten B and V
                    cls_tokens_flat = tf.reshape(cls_tokens, [-1, CONFIG.hidden_dim])
                    
                    # Take the first 2 dimensions for plotting (as per your original logic)
                    cls_2d = cls_tokens_flat[:, :2]
                    cls_2d_global = cls_2d[:NUM_GLOBAL_VIEWS, :]  # Take global views of a sample
                    cls_2d_local = cls_2d[NUM_GLOBAL_VIEWS:TOTAL_VIEWS, :]  # Local views of a sample
                    cls_2d_remaining = cls_2d[TOTAL_VIEWS:, :]  # Remaining samples
                    
                    cls_2d_np = tf.cast(cls_2d, tf.float32).numpy()
                    cls_2d_global = tf.cast(cls_2d_global, tf.float32).numpy()
                    cls_2d_local = tf.cast(cls_2d_local, tf.float32).numpy()
                    cls_2d_remaining = tf.cast(cls_2d_remaining, tf.float32).numpy()

                    plt.figure(figsize=(6,6))
                    plt.scatter(cls_2d_global[:,0], cls_2d_global[:,1], alpha=0.7, label="Sample Global Views", color='red')
                    plt.scatter(cls_2d_local[:,0], cls_2d_local[:,1], alpha=0.5, label="Sample Local Views", color='purple')
                    plt.scatter(cls_2d_remaining[:,0], cls_2d_remaining[:,1], alpha=0.2, color='blue')
                    plt.legend()
                    plt.title(f"CLS Token Distribution - E {epoch+1} S {step} - SIGReg {float(sigreg_loss_backbone):.4f} - Std ({float(np.std(cls_2d_np[:,0])):.2f}, {float(np.std(cls_2d_np[:,1])):.2f})")
                    plt.xlabel("Dimension 1")
                    plt.ylabel("Dimension 2")
                    # plt.show()
                    plt.savefig(f"figures/cls_token_distribution_epoch{epoch+1}_step{step}.png")
                    wandb.log({
                        "cls_token_distribution": wandb.Image(plt),
                        "epoch": epoch,
                        "step": step
                    })
                    plt.close()
                except Exception as e:
                    print(f"Skipping visualization due to error during visualization at step {step}: {e}")
        avg_loss_backbone = total_loss_backbone / num_batches if num_batches > 0 else 0.0
        avg_loss_probe = total_loss_probe / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s.")
        print(f"  Avg Backbone Loss: {avg_loss_backbone:.4f}")
        print(f"  Avg Probe Loss:    {avg_loss_probe:.4f}")
        print(f"  Avg Probe Accuracy: {avg_acc:.4f}")
        
        # Best model saving 
        current_loss = avg_loss_backbone
        
        if current_loss < best_loss_var:
            print(f"  [IMPROVEMENT] Loss improved from {best_loss_var.numpy():.4f} to {current_loss:.4f}!")
            best_loss_var.assign(current_loss)
            # Save to the 'best' directory with explicit epoch number
            save_path = best_checkpoint_manager.save(checkpoint_number=epoch+1)
            print(f"  Saved Best Model to: {save_path}")
        else:
            print(f"  [INFO] Loss {current_loss:.4f} did not beat best {best_loss_var.numpy():.4f}.")
        # --------------------------------------

        wandb.log({
            "epoch_avg_backbone_loss": avg_loss_backbone,
            "epoch_avg_probe_loss": avg_loss_probe,
            "epoch_avg_probe_accuracy": avg_acc,
            "best_loss": best_loss_var.numpy(),
            "epoch": epoch
        })
        
        # Save checkpoint
        checkpoint_manager.save()

if __name__ == "__main__":
    main()
