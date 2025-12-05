import tensorflow as tf

class LinearProbe(tf.keras.Model):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.dense = tf.keras.layers.Dense(num_classes, input_shape=(input_dim,))

    def call(self, x):
        return self.dense(x)