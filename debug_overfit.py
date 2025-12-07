import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.transformer import ViT_Ti
from models.probe import LinearProbe
from data.spec_dataset import get_dataset
from constants import *
from train import balanced_acc

# Force eager execution for easier debugging if needed, though graph is faster
# tf.config.run_functions_eagerly(True)

def debug_overfit():
    print("Setting up debug overfitting experiment...")
    
    # 1. Data Setup
    # We use the same dataset loader but will only take ONE batch
    ds = get_dataset(
        data_dir=DATA_DIR, 
        csv_path=CSV_PATH,
        batch_size=GLOBAL_BATCH_SIZE,
        training=False # No augmentation for deterministic overfitting
    )
    
    # Take a single batch
    for batch_views, batch_labels in ds.take(1):
        # batch_views: (B, V, H, W, C) -> We only need one view for supervised check
        # batch_labels: (B, V, NumClasses) -> We only need one set of labels
        
        # Use the first view (global view)
        inputs = batch_views[:, 0, :, :, :] # (B, H, W, C)
        inputs = tf.expand_dims(inputs, axis=1) # (B, 1, H, W, C)
        labels = batch_labels[:, 0, :]      # (B, NumClasses)
        break
    
    print(f"Loaded single batch. Inputs: {inputs.shape}, Labels: {labels.shape}")
    
    # Check stats
    print(f"Input Stats - Min: {tf.reduce_min(inputs):.4f}, Max: {tf.reduce_max(inputs):.4f}, Mean: {tf.reduce_mean(inputs):.4f}")
    print(f"Label Stats - Max: {tf.reduce_max(labels)}, Sum per sample: {tf.reduce_mean(tf.reduce_sum(labels, axis=1))}")

    # 2. Model Setup
    model = ViT_Ti(CONFIG)
    probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)
    
    # Build models
    _ = model(inputs[:1], training=True)
    _ = probe(tf.zeros((1, CONFIG.hidden_dim)), training=True)
    
    # 3. Optimizer & Loss
    # Use SGD with high momentum to force movement
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    print("\nStarting training loop (100 epochs)...")
    
    losses = []
    accuracies = []
    
    # Reduce to 4 samples to make it super easy
    inputs = inputs[:4]
    labels = labels[:4]
    print(f"Reduced to 4 samples for strict overfitting check.")
    
    for epoch in range(100):
        with tf.GradientTape() as tape:
            # Forward pass
            # ViT outputs: (B, V, Seq, D) -> We want CLS token
            outputs = model(inputs, training=True) 
            
            # outputs[:, 0, 0, :] -> View 0, Token 0 (CLS)
            cls_token = outputs[:, 0, 0, :] # (B, D)
            
            # Probe
            logits = probe(cls_token, training=True) # (B, NumClasses)
            
            # Compute Loss
            loss = criterion(labels, logits)
            
        # Compute Gradients
        trainable_vars = model.trainable_variables + probe.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Check Gradient Norms
        if epoch % 10 == 0:
            grad_norm_model = tf.linalg.global_norm(gradients[:len(model.trainable_variables)])
            grad_norm_probe = tf.linalg.global_norm(gradients[len(model.trainable_variables):])
            print(f"  Grad Norm Model: {float(grad_norm_model):.4f}, Probe: {float(grad_norm_probe):.4f}")
        
        # Update Weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Metrics
        probs = tf.nn.sigmoid(logits)
        acc = balanced_acc(labels, probs)
        
        losses.append(loss.numpy())
        accuracies.append(acc.numpy())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {float(loss):.4f}, Balanced Acc = {float(acc):.4f}")
            # Check embedding std to ensure no collapse
            emb_std = tf.math.reduce_std(cls_token, axis=0)
            print(f"  Embed Batch Std: {float(tf.reduce_mean(emb_std)):.4f}")

    print(f"Final Epoch: Loss = {float(loss):.4f}, Balanced Acc = {float(acc):.4f}")
    
    # 4. Final Evaluation
    print("\n========== FINAL PREDICTIONS (First 3 Samples) ==========")
    for i in range(3):
        true_inds = tf.where(labels[i] == 1)[:, 0].numpy()
        pred_probs, pred_inds = tf.math.top_k(probs[i], k=5)
        
        print(f"Sample {i}:")
        print(f"  True Indices: {true_inds}")
        print(f"  Pred Indices: {pred_inds.numpy()}")
        print(f"  Pred Probs:   {pred_probs.numpy()}")
    print("=========================================================")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Balanced Accuracy")
    plt.savefig("debug_overfit_curve.png")
    print("Saved training curve to debug_overfit_curve.png")

if __name__ == "__main__":
    debug_overfit()
