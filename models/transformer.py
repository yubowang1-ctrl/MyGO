from dataclasses import dataclass
import random
import tensorflow as tf 
import numpy as np
from constants import ViTConfig

def gen_maskid_patch(batch_size, num_col_patches, num_row_patches, G, V, num_mask=200, cluster_size=3):
    """Generates random mask indices for patches.
    Args:
        batch_size: Number of samples in the batch
        num_col_patches: Number of patches along the columns (frequency dimension)
        num_row_patches: Number of patches along the rows (time dimension)
        num_mask: Number of patches to mask
        cluster_size: Size of each mask cluster
    Returns:
        A boolean mask of shape (batch_size, seq_len) where True indicates a masked patch.
    """
    
    num_col_patches = tf.cast(num_col_patches, tf.int32)
    num_row_patches = tf.cast(num_row_patches, tf.int32)
    seq_len = num_col_patches * num_row_patches
    
    # 1. Determine which samples are global vs local
    batch_indices = tf.range(batch_size)
    is_local = (batch_indices % V) >= G # (batch_size,) boolean
    
    # 2. Generate masks for ALL samples assuming they are local
    # We simplify the cluster generation to be vectorizable: fixed cluster size for the whole batch
    # or we just generate random centers and scatter.
    
    # To keep it simple and vectorizable:
    # We generate random centers for the whole batch at once.
    # We assume a fixed cluster size for vectorization simplicity (or max size).
    
    # Number of clusters per sample
    est_clusters = tf.cast(tf.math.ceil(tf.cast(num_mask, tf.float32) / (float(cluster_size)**2)), tf.int32) + 2
    
    # Generate centers for all samples: (batch_size, est_clusters)
    center_indices = tf.random.uniform([batch_size, est_clusters], minval=0, maxval=seq_len, dtype=tf.int32)
    
    # Create offsets for a single cluster
    range_vec = tf.range(cluster_size)
    grid_i, grid_j = tf.meshgrid(range_vec, range_vec)
    grid_i = tf.reshape(grid_i, [-1])
    grid_j = tf.reshape(grid_j, [-1])
    offsets = grid_i * num_col_patches + grid_j # (cluster_size^2,)
    
    # Broadcast add to get all patch indices for all clusters
    # center_indices: (B, K) -> (B, K, 1)
    # offsets: (M,) -> (1, 1, M)
    # Result: (B, K, M)
    cluster_indices = tf.expand_dims(center_indices, -1) + tf.reshape(offsets, [1, 1, -1])
    cluster_indices = tf.reshape(cluster_indices, [batch_size, -1]) # (B, K*M)
    
    # Indices must be (Num_Updates, 2) where 2 is (batch_idx, seq_idx)
    
    B_grid = tf.expand_dims(tf.range(batch_size), -1) # (B, 1)
    B_grid = tf.tile(B_grid, [1, tf.shape(cluster_indices)[1]]) # (B, K*M)
    
    flat_batch_indices = tf.reshape(B_grid, [-1])
    flat_seq_indices = tf.reshape(cluster_indices, [-1])
    
    # Filter valid indices
    valid_mask = (flat_seq_indices >= 0) & (flat_seq_indices < seq_len)
    
    valid_batch_indices = tf.boolean_mask(flat_batch_indices, valid_mask)
    valid_seq_indices = tf.boolean_mask(flat_seq_indices, valid_mask)

    valid_batch_indices = tf.clip_by_value(valid_batch_indices, 0, batch_size-1)
    valid_seq_indices = tf.clip_by_value(valid_seq_indices, 0, seq_len - 1)
    
    scatter_indices = tf.stack([valid_batch_indices, valid_seq_indices], axis=1) # (N_valid, 2)
    updates = tf.ones([tf.shape(scatter_indices)[0]], dtype=tf.bool)
    
    # Start with all False
    batch_mask_local = tf.scatter_nd(scatter_indices, updates, [batch_size, seq_len])
    
    # 3. Combine Global (all False) and Local masks
    # Global samples should be all False.
    # We can just multiply the generated mask by is_local
    
    is_local_expanded = tf.expand_dims(is_local, -1) # (B, 1)
    batch_mask = batch_mask_local & is_local_expanded
    
    return batch_mask

def mask_patches(batch_size, num_row_patches, num_col_patches, G, V):
    """
    Mask patches in the input tensor x according to the boolean mask.
    """
    mask = gen_maskid_patch(batch_size, num_col_patches, num_row_patches, G, V)
    mask = tf.expand_dims(mask, axis=-1)  # shape (batch_size, seq_len, 1)
    return mask
    
def mask_timeframe(batch_size, num_row_patches, num_col_patches, G, V, ratio=0.5):
    """
    Mask entire time frames (colums of patches) in the input tensor x.
    
    Note: Ratio only applies to local views.
    """
    # Ensure inputs are tensors
    batch_size = tf.cast(batch_size, tf.int32)
    num_row_patches = tf.cast(num_row_patches, tf.int32)
    num_col_patches = tf.cast(num_col_patches, tf.int32)
    
    num_time_frames = num_col_patches
    # Use tf.cast for float calculation then back to int
    num_unmask = tf.cast(tf.cast(num_time_frames, tf.float32) * (1.0 - ratio), tf.int32)
    
    # 1. Determine which samples are global vs local
    batch_indices = tf.range(batch_size)
    is_local = (batch_indices % V) >= G # (batch_size,) boolean
    
    # 2. Generate masks for ALL samples assuming they are local
    # We want to mask everything EXCEPT a window of size num_unmask
    # So we start with all True (Masked) and set a window to False (Unmasked)
    
    max_start = num_time_frames - num_unmask
    # Generate random start for each sample in batch
    start_frames = tf.random.uniform([batch_size], minval=0, maxval=max_start + 1, dtype=tf.int32) # (B,)
    
    # Create a grid of column indices: (1, num_time_frames)
    col_indices = tf.range(num_time_frames)
    col_indices = tf.expand_dims(col_indices, 0) # (1, T)
    
    # Expand start_frames: (B, 1)
    start_frames_expanded = tf.expand_dims(start_frames, -1)
    
    # Check if index is within [start, start + num_unmask)
    # These are the UNMASKED indices.
    is_unmasked = (col_indices >= start_frames_expanded) & (col_indices < (start_frames_expanded + num_unmask))
    
    # The mask is the inverse: True where masked
    mask_time_frames_local = tf.logical_not(is_unmasked) # (B, T)
    
    # 3. Combine Global (all False) and Local masks
    # Global samples should be all False (no masking).
    is_local_expanded = tf.expand_dims(is_local, -1) # (B, 1)
    mask_time_frames = mask_time_frames_local & is_local_expanded
    
    # Expand to patch level
    # mask_time_frames: (batch_size, num_col_patches)
    mask_expanded = tf.tile(mask_time_frames, [1, num_row_patches])
    # mask_expanded: (batch_size, num_col_patches * num_row_patches)
    full_mask = tf.expand_dims(mask_expanded, axis=-1) # (B, Seq, 1)
    return full_mask

class ViT(tf.keras.Model):
    def __init__(self, config: ViTConfig, **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.config = config
        self.linear_projection = tf.keras.layers.Dense(config.hidden_dim)
        self.transformer_blocks = [
            TransformerBlock(config, name=f"transformer_block_{i}") for i in range(config.num_layers)
        ]
        self.cls_token = self.add_weight( # trainable cls token
            shape=(1, 1, config.hidden_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
        self.mask_token = self.add_weight( # trainable mask token
            shape=(1, 1, config.hidden_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="mask_token",
        )
        
        # validate patching parameters
        ph, pw, po = self.config.patch_height, self.config.patch_width, self.config.patch_overlap
        assert (self.config.image_height - po) % (ph - po) == 0, "Image height not compatible with patch height and overlap"
        assert (self.config.image_width - po) % (pw - po) == 0, "Image width not compatible with patch width and overlap"   
        assert po < ph and po < pw, "Patch overlap must be smaller than patch dimensions"
        assert self.config.G > 0 and self.config.V > 0 and self.config.G <= self.config.V, "Number of global views G must be positive and less than total views V"
        
    def call(self, x, training):
        # x shape: (batch_size, view_number, image_height, image_width, num_channels)
        V = tf.shape(x)[1]  # number of views
        if not training:
            V = 1  # during inference, only process one view at a time
            G = 1
            tf.debugging.assert_equal(tf.shape(x)[1], 1, "During inference, input must have exactly one view.")
        x = tf.reshape(x, [-1, self.config.image_height, self.config.image_width, self.config.num_channels])
        
        # Now, cut into patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.config.patch_height, self.config.patch_width, 1],
            strides=[1, self.config.patch_height - self.config.patch_overlap, self.config.patch_width - self.config.patch_overlap, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        batch_size = tf.shape(patches)[0]
        num_row_patches = tf.shape(patches)[1]
        num_col_patches = tf.shape(patches)[2]
        # tf.print(tf.shape(patches))
        # patches now has shape (batch_size * V, num_row_patches, num_col_patches, patch_height * patch_width * num_channels)
        
        # Ordering of patches matter!
        # This is a spectrogram image:
        # [ [patch_1, patch_2, ..., patch_n],
        #   [patch_(n+1), patch_(n+2), ..., patch_(2n)],
        #   ...
        #   [patch_((m-1)*n+1), ..., patch_(m*n)] ]
        # --> time goes along rows, frequency along columns
        # where each patch is of shape (patch_height, patch_width, num_channels)
        
        x = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1] * tf.shape(patches)[2], tf.shape(patches)[3]])  # flatten patches
        x = self.linear_projection(x)  # project to hidden_dim
        # x shape: (batch_size * V, num_patches, hidden_dim)
        if training: # mask patches
            mask = mask_patches(batch_size, num_row_patches, num_col_patches, self.config.G, self.config.V)
            mask |= mask_timeframe(batch_size, num_row_patches, num_col_patches, self.config.G, self.config.V)
            x = tf.where(mask, tf.broadcast_to(self.mask_token, tf.shape(x)), x)
            
        # prepend cls token to the sequence
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.config.hidden_dim])
        x = tf.concat([cls_tokens, x], axis=1)  # (batch_size, num_patches + 1, hidden_dim)
        # pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        # x is now (batch_size*V, num_patches + 1, hidden_dim)
        
        # Reshape back to (batch_size, view_number, num_patches + 1, hidden_dim)
        x = tf.reshape(x, [-1, V, tf.shape(x)[1], tf.shape(x)[2]])
        return x  # return all embeddings including cls token embeddings

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.self_attention = SelfAttention(
            config,
            initializer=tf.keras.initializers.HeUniform()
        )
        self.mlp_block = MLPBlock(
            config,
            initializer=tf.keras.initializers.HeUniform()
        )
    
    def call(self, x, training):
        # just a self-attention layer followed by MLP block
        # layernorms and residual connections are handled inside those layers
        x = self.self_attention(x, training=training)
        x = self.mlp_block(x, training=training)
        return x

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, initializer, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(config, initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        attn_output = self.mha(x, training=training) # multihead self-attention
        attn_output = self.dropout(attn_output, training=training) # set random embeddings to 0
        out = self.layernorm(x + attn_output) # residual connection, then layernorm
        return out

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, initializer, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(config.mlp_dim, activation='gelu', kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(config.hidden_dim, kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        mlp_output = self.dense1(x)
        mlp_output = self.dropout(mlp_output, training=training)
        mlp_output = self.dense2(mlp_output)
        mlp_output = self.dropout(mlp_output, training=training)
        out = self.layernorm(x + mlp_output) # residual connection, then layernorm
        return out

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, initializer, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.key_dim = config.hidden_dim // config.num_heads # key_dim is for per head, so hidden_dim = num_heads * key_dim
        self.heads = [SingleHeadAttention(self.key_dim, config, initializer=initializer, name=f"head_{i}") for i in range(config.num_heads)]
        self.num_heads = config.num_heads
        self.output_dense = tf.keras.layers.Dense(config.hidden_dim, kernel_initializer=initializer)
    
    def call(self, embeddings, training):
        head_outputs = [head(embeddings, training=training) for head in self.heads]
        concat_heads = tf.concat(head_outputs, axis=-1) # concatenate on the last dimension (features)
        output = self.output_dense(concat_heads)
        # dropout for W_O is handled in SelfAttention class
        return output

class SingleHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, config: ViTConfig, initializer, **kwargs):
        super(SingleHeadAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.query_dense = tf.keras.layers.Dense(key_dim, kernel_initializer=initializer)
        self.key_dense = tf.keras.layers.Dense(key_dim, kernel_initializer=initializer)
        self.value_dense = tf.keras.layers.Dense(key_dim, kernel_initializer=initializer)
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout_rate)
        
        self.num_patch_per_col = (config.image_height - config.patch_overlap) // (config.patch_height - config.patch_overlap)
        self.num_patch_per_row = (config.image_width - config.patch_overlap) // (config.patch_width - config.patch_overlap)

        self.max_time_distance = 2 * self.num_patch_per_col - 1 # relative position bias for time dimension
        self.attention_bias_table = self.add_weight(
            shape=(2 * self.max_time_distance - 1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="time_attention_bias_table",
        )
        self.cls_token_bias_table = self.add_weight(
            shape=(self.num_patch_per_col * self.num_patch_per_row + 1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="cls_token_bias_table",
        )
        
    def call(self, embeddings, training):
        Q = self.query_dense(embeddings) # (batch_size, seq_len, key_dim)
        K = self.key_dense(embeddings)   # (batch_size, seq_len, key_dim)
        V = self.value_dense(embeddings) # (batch_size, seq_len, key_dim)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(Q, K, transpose_b=True) # (batch_size, seq_len, seq_len)
        dk = tf.cast(self.key_dim, embeddings.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Add relative position bias
        # class embedding receives bias normally as other embeddings
        # Assuming seq_len is always fixed during training and inference
        # So we use max_time_distance to build the relative position bias table
        
        position_indices = tf.range(self.num_patch_per_row)
        relative_positions = position_indices[None, :] - position_indices[:, None] # generate position difference matrix
        clipped_relative_positions = tf.clip_by_value(relative_positions, -self.max_time_distance + 1, self.max_time_distance - 1) # prevent out-of-bounds
        bias_indices = clipped_relative_positions + self.max_time_distance - 1 # shift to non-negative
        relative_time_bias = tf.gather(self.attention_bias_table, bias_indices) # (num_patch_per_row, num_patch_per_row)
        # Expand by repeating in both dimensions to match seq_len
        relative_position_bias = tf.repeat(relative_time_bias, repeats=self.num_patch_per_col, axis=0)
        relative_position_bias = tf.repeat(relative_position_bias, repeats=self.num_patch_per_col, axis=1)
        # Now shape is (seq_len - 1, seq_len - 1), need to add in class token bias table
        # Add class token bias to first row and first column
        # Expand dims of cls_token_bias_table[1:] to (1, N) to match relative_position_bias (N, N)
        cls_row = tf.expand_dims(self.cls_token_bias_table[1:], 0)
        relative_time_bias = tf.concat(
            [self.cls_token_bias_table[:,None], tf.concat([cls_row, relative_position_bias], axis=0)],
            axis=1
        ) # (seq_len, seq_len)
        
        scaled_attention_logits += relative_time_bias
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (batch_size, seq_len, seq_len)
        attention_weights = self.attention_dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, V) # (batch_size, seq_len, key_dim)
        return output