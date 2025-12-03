import os
import time
import tensorflow as tf
from models.transformer import ViT, ViTConfig
from models.loss import LeJEPA
from data.dataset import get_dataset
from constants import *

# ==============================================================================
# 2. Setup Distributed Strategy
# ==============================================================================
# MirroredStrategy handles both single-GPU and multi-GPU scenarios.
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# ==============================================================================
# 3. Data Loading (Placeholder)
# ==============================================================================
def get_dataset():
    """
    Placeholder for data loading logic.
    Should return a tf.data.Dataset yielding batches of shape:
    (GLOBAL_BATCH_SIZE, TOTAL_VIEWS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    """
    # TODO: Replace this with actual data loading from .npy files or TFRecords
    # Example:
    # dataset = tf.data.Dataset.list_files("data/*.npy")
    # dataset = dataset.map(load_and_preprocess_function)
    
    # Dummy data for demonstration
    dummy_images = tf.random.normal(
        (1000, TOTAL_VIEWS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    )
    dataset = tf.data.Dataset.from_tensor_slices(dummy_images)
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create and distribute the dataset
# experimental_distribute_dataset splits the global batch among replicas
with strategy.scope():
    dataset = get_dataset()
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

# ==============================================================================
# 4. Model, Optimizer, and Loss Initialization
# ==============================================================================
with strategy.scope():
    # Initialize Model
    model = ViT(CONFIG)
    
    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Initialize Loss
    loss_fn = LeJEPA(G=NUM_GLOBAL_VIEWS, V=TOTAL_VIEWS)
    
    # Checkpoint Manager (Optional but recommended)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=5
    )

# ==============================================================================
# 5. Training Step
# ==============================================================================
def train_step(inputs):
    """
    Runs on EACH replica.
    inputs: (Per_Replica_Batch_Size, TOTAL_VIEWS, H, W, C)
    """
    # 1. Flatten views into the batch dimension for the model
    # Shape: (B * V, H, W, C)
    B = tf.shape(inputs)[0]
    V = TOTAL_VIEWS
    
    flat_inputs = tf.reshape(inputs, [B * V, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    
    with tf.GradientTape() as tape:
        # 2. Forward Pass
        # Output: (B * V, Seq_Len, Hidden_Dim)
        outputs = model(flat_inputs, training=True)
        
        # 3. Extract CLS Token (Index 0)
        # Shape: (B * V, Hidden_Dim)
        cls_tokens = outputs[:, 0, :]
        
        # 4. Prepare inputs for LeJEPA Loss
        # all_emb: (B * V, D) -> This is just cls_tokens
        all_emb = cls_tokens
        
        # global_emb: Needs to be (B * G, D)
        # Reshape to (B, V, D) to slice views
        cls_reshaped = tf.reshape(cls_tokens, [B, V, -1])
        
        # Slice the first G views (assuming they are the global ones)
        global_views = cls_reshaped[:, :NUM_GLOBAL_VIEWS, :] # (B, G, D)
        
        # Flatten back to (B * G, D)
        global_emb = tf.reshape(global_views, [B * NUM_GLOBAL_VIEWS, -1])
        
        # 5. Calculate Loss
        # LeJEPA returns a scalar (sum of inv_loss and sigreg_loss)
        # Note: The internal reduce_mean in LeJEPA acts on the local batch.
        per_replica_loss = loss_fn(global_emb, all_emb, lambd=LAMBDA_SIGREG)
        
        # 6. Scale Loss for Global Batch
        # We divide by the number of replicas so that when we sum gradients (implicit)
        # or sum losses (explicit), we get the correct global average.
        scaled_loss = per_replica_loss / float(strategy.num_replicas_in_sync)

    # 7. Compute and Apply Gradients
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return scaled_loss

# ==============================================================================
# 6. Distributed Training Loop
# ==============================================================================
@tf.function
def distributed_train_step(dataset_inputs):
    # Run train_step on all replicas
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    
    # Aggregate losses (SUM) to get the global average loss
    # (Since we divided by num_replicas inside train_step)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def main():
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        start_time = time.time()
        
        total_loss = 0.0
        num_batches = 0
        
        # Iterate over the distributed dataset
        for step, batch_inputs in enumerate(dist_dataset):
            loss = distributed_train_step(batch_inputs)
            
            total_loss += loss
            num_batches += 1
            
            if step % LOG_EVERY_STEPS == 0:
                print(f"  Step {step}: Loss = {loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_manager.save()

if __name__ == "__main__":
    main()
