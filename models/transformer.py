from dataclasses import dataclass
import tensorflow as tf 

@dataclass
class ViTConfig:
    image_height: int
    image_width: int
    num_channels: int
    patch_height: int
    patch_width: int
    num_layers: int
    hidden_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float
    max_token_length: int # required for relative time embeddings
    
# Tiny ViT with 16x16 patches, ~5M parameters
ViT-Ti-16 = ViTConfig(
    image_height=224,
    image_width=224,
    num_channels=3,
    patch_height=16,
    patch_width=16,
    num_layers=12,
    hidden_dim=192,
    mlp_dim=768,
    num_heads=3,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    max_token_length=197, # 14*14 + 1 for CLS token
) 

# Small ViT with 16x16 patches, ~22M parameters
ViT-S-16 = ViTConfig(
    image_height=224,
    image_width=224,
    num_channels=3,
    patch_height=16,
    patch_width=16,
    num_layers=12,
    hidden_dim=384,
    mlp_dim=1536,
    num_heads=6,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    max_token_length=197, # 14*14 + 1 for CLS token
) 

# Base ViT with 16x16 patches, ~86M parameters
ViT-B-16 = ViTConfig(
    image_height=224,
    image_width=224,
    num_channels=3,
    patch_height=16,
    patch_width=16,
    num_layers=12,
    hidden_dim=768,
    mlp_dim=3072,
    num_heads=12,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    max_token_length=197, # 14*14 + 1 for CLS token
)

class ViT(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.config = config
        self.linear_projection = tf.keras.layers.Dense(config.hidden_dim)
        self.transformer_blocks = [
            TransformerBlock(config, name=f"transformer_block_{i}") for i in range(config.num_layers)
        ]
        self.cls_token = self.add_weight(
            shape=(1, 1, config.hidden_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
    
    def call(self, x, training):
        # x shape: (batch_size, num_patches, image_height, image_width, num_channels)
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])  # flatten patches
        x = self.linear_projection(x)  # project to hidden_dim
        # prepend cls token to the sequence
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.config.hidden_dim])
        x = tf.concat([cls_tokens, x], axis=1)  # (batch_size, num_patches + 1, hidden_dim)
        # pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return x # (batch_size, num_patches + 1, hidden_dim)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.self_attention = SelfAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            max_token_length=config.max_token_length,
            initializer=tf.keras.initializers.HeUniform()
        )
        self.mlp_block = MLPBlock(
            hidden_dim=config.hidden_dim,
            mlp_dim=config.mlp_dim,
            dropout_rate=config.dropout_rate,
            initializer=tf.keras.initializers.HeUniform()
        )
    
    def call(self, x, training):
        # just a self-attention layer followed by MLP block
        # layernorms and residual connections are handled inside those layers
        x = self.self_attention(x, training=training)
        x = self.mlp_block(x, training=training)
        return x

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, dropout_rate, attention_dropout_rate, max_token_length, initializer, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads, attention_dropout_rate=attention_dropout_rate, max_token_length=max_token_length, initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        attn_output = self.mha(x, x) # multihead self-attention
        attn_output = self.dropout(attn_output, training=training) # set random embeddings to 0
        out = self.layernorm(x + attn_output) # residual connection, then layernorm
        return out

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, mlp_dim, dropout_rate, initializer, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(mlp_dim, activation='gelu', kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(hidden_dim, kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        mlp_output = self.dense1(x)
        mlp_output = self.dropout(mlp_output, training=training)
        mlp_output = self.dense2(mlp_output)
        mlp_output = self.dropout(mlp_output, training=training)
        out = self.layernorm(x + mlp_output) # residual connection, then layernorm
        return out

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, attention_dropout_rate, max_token_length, initializer, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = [SingleHeadAttention(key_dim, attention_dropout_rate, max_token_length, initializer=initializer, name=f"head_{i}") for i in range(num_heads)]
        self.num_heads = num_heads
        self.key_dim = key_dim # key_dim is for per head, so hidden_dim = num_heads * key_dim
        self.output_dense = tf.keras.layers.Dense(num_heads * key_dim, kernel_initializer=initializer)
    
    def call(self, embeddings, training):
        head_outputs = [head(embeddings, training=training) for head in self.heads]
        concat_heads = tf.concat(head_outputs, axis=-1) # concatenate on the last dimension (features)
        output = self.output_dense(concat_heads)
        # dropout for W_O is handled in SelfAttention class
        return output

class SingleHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, attention_dropout_rate, max_token_length, initializer, **kwargs):
        super(SingleHeadAttention, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.query_dense = tf.keras.layers.Dense(key_dim, kernel_initializer=initializer)
        self.key_dense = tf.keras.layers.Dense(key_dim, kernel_initializer=initializer)
        self.value_dense = tf.keras.layers.Dense(key_dim, kernel_initializer=initializer)
        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)
        self.max_token_length = max_token_length
        self.attention_bias_table = self.add_weight(
            shape=(2 * max_token_length - 1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="attention_bias_table",
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
        seq_len = tf.shape(embeddings)[1]
        position_indices = tf.range(seq_len)
        relative_positions = position_indices[None, :] - position_indices[:, None] # (seq_len, seq_len) generate position difference matrix
        clipped_relative_positions = tf.clip_by_value(relative_positions, -self.max_token_length + 1, self.max_token_length - 1) # prevent out-of-bounds
        bias_indices = clipped_relative_positions + self.max_token_length - 1 # shift to non-negative
        relative_position_bias = tf.gather(self.attention_bias_table, bias_indices) # (seq_len, seq_len)
        attention_weights += relative_position_bias
        
        attention_weights = self.attention_dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, V) # (batch_size, seq_len, key_dim)
        return output