import tensorflow as tf
import math

def all_reduce(x, op="MEAN"):
    """
    Aggregates tensors across all devices in a distributed setting.
    """
    # Check if we are in a distributed strategy context
    if tf.distribute.has_strategy():
        ctx = tf.distribute.get_replica_context()
        if ctx is not None:
            # Map string op to TF ReduceOp
            reduce_op = tf.distribute.ReduceOp.MEAN
            if op.upper() == "SUM":
                reduce_op = tf.distribute.ReduceOp.SUM
            
            return ctx.all_reduce(reduce_op, x)
    
    # If not distributed, just return the input
    return x

class EppsPulley(tf.keras.layers.Layer):
    def __init__(self, t_max=3.0, n_points=17, integration="trapezoid", **kwargs):
        super().__init__(**kwargs)
        if n_points % 2 == 0:
            raise ValueError("n_points must be odd")
            
        self.n_points = n_points
        self.integration = integration
        
        # 1. Create the grid 't'
        # Linearly spaced positive points (including 0)
        t = tf.linspace(0.0, t_max, n_points)
        self.t = tf.cast(t, dtype=tf.float32)
        
        # 2. Precompute weights
        dt = t_max / (n_points - 1)
        
        # Create weights tensor
        weights_list = [2 * dt] * n_points
        weights_list[0] = dt
        weights_list[-1] = dt
        weights = tf.constant(weights_list, dtype=tf.float32)
        
        # 3. Precompute Phi (Standard Normal CF)
        # phi = exp(-t^2 / 2)
        self.phi = tf.exp(-0.5 * tf.square(self.t))
        
        # Combine weights and phi
        self.weighted_phi = weights * self.phi

    def call(self, x):
        # x shape: (Batch_Size, K) where K is number of slices/features
        # We want to compute the test statistic for each slice independently.
        
        # Get N (Batch Size)
        N = tf.cast(tf.shape(x)[0], tf.float32)
        
        # Expand dims to broadcast with t
        # x: (Batch, K, 1)
        x_expanded = tf.expand_dims(x, -1)
        
        # t: (1, 1, n_points)
        t_expanded = tf.reshape(self.t, [1, 1, -1])
        
        # x_t: (Batch, K, n_points)
        x_t = x_expanded * t_expanded
        
        cos_vals = tf.cos(x_t)
        sin_vals = tf.sin(x_t)
        
        # Mean across batch (axis 0) -> (K, n_points)
        cos_mean = tf.reduce_mean(cos_vals, axis=0)
        sin_mean = tf.reduce_mean(sin_vals, axis=0)
        
        # --- Distributed Step ---
        # If running on multiple GPUs, average these means across all GPUs
        cos_mean = all_reduce(cos_mean, op="MEAN")
        sin_mean = all_reduce(sin_mean, op="MEAN")
        
        # Compute error
        # (Empirical_Real - Normal_Real)^2 + (Empirical_Imag)^2
        # Note: Normal_Imag is 0 because standard normal is symmetric
        # self.phi is (n_points,) -> broadcast to (K, n_points)
        err = tf.square(cos_mean - self.phi) + tf.square(sin_mean)
        
        # Weighted integration
        # Dot product of error and weights along the last axis (n_points)
        # err: (K, n_points), weighted_phi: (n_points,)
        # Result: (K,)
        integral = tf.tensordot(err, self.weighted_phi, axes=[[1], [0]])
        
        # Get world size (number of GPUs) to scale N correctly
        strategy = tf.distribute.get_strategy()
        world_size = tf.cast(strategy.num_replicas_in_sync, tf.float32)
        
        # Return statistic per slice
        return integral * N * world_size


class SIGReg(tf.keras.losses.Loss):
    def __init__(self, name="SIGReg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epps_pulley = EppsPulley()

    def call(self, x, global_step, num_slices=256):
        """
        Args:
            x: Input tensor of shape (Batch, D)
            global_step: Tensor or integer for random seed synchronization
            num_slices: Number of random slices to project onto
        """
        D = tf.shape(x)[-1]
        
        # Ensure global_step is a tensor for consistency
        global_step = tf.convert_to_tensor(global_step)
        
        # Synchronize global_step across replicas to ensure same seed
        # We use MEAN and cast back to int, assuming all replicas have same step roughly
        global_step_float = tf.cast(global_step, tf.float32)
        global_step_sync = all_reduce(global_step_float, op="MEAN")
        seed = tf.cast(global_step_sync, tf.int32)
        
        # Generate random projection matrix A: (D, num_slices)
        # We use stateless_normal to ensure determinism with the seed
        A = tf.random.stateless_normal(
            shape=[D, num_slices], 
            seed=[seed, 0],
            dtype=tf.float32
        )
        
        # Normalize columns of A to be unit vectors
        norm = tf.norm(A, axis=0, keepdims=True)
        A = A / (norm + 1e-8)
        
        # Project x onto A
        # x: (Batch, D) @ A: (D, num_slices) -> (Batch, num_slices)
        x_proj = tf.matmul(x, A)
        
        # Apply univariate test on each slice
        # stats: (num_slices,)
        stats = self.epps_pulley(x_proj)
        
        # Aggregate results (mean over slices)
        return tf.reduce_mean(stats)
    
    
class LeJEPA(tf.keras.losses.Loss):
    def __init__(self, G, V, name="LeJEPA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.V = V # number of views for each sample
        self.G = G # number of global views for each sample
        self.sigreg = SIGReg()
    
    def call(self, global_emb, all_emb, lambd):
        # global_emb has shape (B*G, D) where B is batch size
        # all_emb has shape (B*V, D) 
        sigreg_loss = self.sigreg(all_emb, tf.cast(tf.compat.v1.train.get_or_create_global_step(), tf.int32))
        
        global_emb = tf.reshape(global_emb, [-1, self.G, tf.shape(global_emb)[-1]])  # (B, G, D)
        all_emb = tf.reshape(all_emb, [-1, self.V, tf.shape(all_emb)[-1]])          # (B, V, D)
        
        global_centers = tf.reduce_mean(global_emb, axis=1, keepdims=True) # (B, 1, D)
        diff = all_emb - global_centers  # (B, V, D)
        inv_loss = tf.reduce_mean(tf.square(diff))  # (scalar)
        
        return inv_loss * (1 - lambd) + sigreg_loss * lambd