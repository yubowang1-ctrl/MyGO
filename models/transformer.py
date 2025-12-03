from dataclasses import dataclass
import random
import tensorflow as tf 
import numpy as np
from constants import ViTConfig

def gen_maskid_patch(batch_size, num_col_patches, num_row_patches, G, V, num_mask=100, cluster_size=3):
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
    
    # We will generate masks for each sample in parallel using tf.map_fn or vectorized ops
    # But generating complex clusters is hard to vectorize perfectly.
    # A simpler approach for TF graph is to generate random centers and expand them.
    
    def generate_single_mask(batch_idx):
        
        if batch_idx % V < G:
            # Global view: No masking
            return tf.zeros([seq_len], dtype=tf.bool)
        
        this_cluster_size = tf.cast(cluster_size, tf.int32)
        this_cluster_size += tf.random.uniform([], minval=-1, maxval=3, dtype=tf.int32)  # add some randomness to cluster size
        # Start with all False
        mask = tf.zeros([seq_len], dtype=tf.bool)
        
        # We need to loop until we have enough masked patches
        # Since we can't easily do a "while" loop with dynamic updates in a simple way,
        # we'll generate a fixed number of clusters that is likely to cover enough area.
        
        # Generate K random centers, where K is roughly num_mask / (this_cluster_size^2)
        # This is an approximation but much faster on GPU.
        
        # Estimate number of clusters needed
        est_clusters = tf.cast(tf.math.ceil(tf.cast(num_mask, tf.float32) / (float(this_cluster_size)**2)), tf.int32)
        # Add some buffer
        est_clusters = est_clusters + 2
        
        # Random centers
        center_indices = tf.random.uniform([est_clusters], minval=0, maxval=seq_len, dtype=tf.int32)
        
        # Expand centers to clusters
        # Create offsets
        offsets = []
        for i in range(this_cluster_size):
            for j in range(this_cluster_size):
                offsets.append(i * num_col_patches + j)
        offsets = tf.stack(offsets) # (this_cluster_size^2,)
        
        # Broadcast add: (est_clusters, 1) + (1, this_cluster_size^2) -> (est_clusters, this_cluster_size^2)
        cluster_indices = tf.expand_dims(center_indices, -1) + tf.expand_dims(offsets, 0)
        cluster_indices = tf.reshape(cluster_indices, [-1])
        
        # Clip to valid range
        valid_indices = tf.boolean_mask(cluster_indices, cluster_indices < seq_len)
        
        # Scatter True to these indices
        # We use tensor_scatter_nd_update
        indices = tf.expand_dims(valid_indices, -1)
        updates = tf.ones_like(valid_indices, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)
        
        return mask

    # Apply to batch
    # Use map_fn to apply to each element in batch
    # dummy = tf.fill([batch_size], [cluster_size])
    # generate batch idx 
    batch_idx = tf.range(batch_size)
    # dummy = tf.stack([dummy, batch_idx], axis=1)  # shape (batch_size, 2)
    dummy = batch_idx  # shape (batch_size,)
    batch_mask = tf.map_fn(generate_single_mask, dummy, fn_output_signature=tf.bool)
    
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
    
    def create_sample_mask(i):
        # Determine if this is a local view
        # i is the index in the batch
        is_local = (i % V) >= G
        if is_local:
            # Create mask: False for unmasked region, True for masked
            # Initialize with all True (Masked)
            mask_row = tf.ones([num_time_frames], dtype=tf.bool)
            
            # Random start for unmasked region
            max_start = num_time_frames - num_unmask
            start_frame = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
            
            # Create indices to unmask
            indices = tf.range(start_frame, start_frame + num_unmask)
            indices = tf.expand_dims(indices, -1)
            
            # Set False (Unmasked) at these indices
            updates = tf.zeros([tf.shape(indices)[0]], dtype=tf.bool)
            mask_row = tf.tensor_scatter_nd_update(mask_row, indices, updates)
            return mask_row
        else:
            # Global view: No masking (all False)
            return tf.zeros([num_time_frames], dtype=tf.bool)

    # Generate for all samples in batch
    # tf.range(batch_size) gives us the indices [0, 1, ... B-1]
    batch_indices = tf.range(batch_size)
    mask_time_frames = tf.map_fn(create_sample_mask, batch_indices, fn_output_signature=tf.bool)
    
    # Expand to patch level
    # mask_time_frames: (batch_size, num_col_patches)
    mask_expanded = tf.tile(mask_time_frames, [1, num_row_patches])
    # mask_expanded: (batch_size, num_col_patches * num_row_patches)
    full_mask = tf.expand_dims(mask_expanded, axis=-1) # (B, Seq, 1)
    return full_mask

class ViT(tf.keras.layers.Layer):
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
        assert self.config.G > 0 and self.config.V > 0 and self.config.G < self.config.V, "Number of global views G must be positive and less than total views V"
        
    def call(self, x, training):
        # x shape: (batch_size, view_number, image_height, image_width, num_channels)
        V = tf.shape(x)[1]  # number of views
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
        attn_output = self.mha(x, x) # multihead self-attention
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
        dk = tf.cast(self.key_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (batch_size, seq_len, seq_len)
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
        relative_time_bias = tf.concat(
            [self.cls_token_bias_table[:,None], tf.concat([self.cls_token_bias_table[1:], relative_position_bias], axis=0)],
            axis=1
        ) # (seq_len, seq_len)
        
        attention_weights += relative_time_bias
        
        attention_weights = self.attention_dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, V) # (batch_size, seq_len, key_dim)
        return output