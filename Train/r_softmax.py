import tensorflow as tf
from tensorflow.keras.layers import Layer


class RSoftmax(Layer):
    def __init__(self, initial_sparsity_rate, **kwargs):
        super(RSoftmax, self).__init__(**kwargs)
        self.initial_sparsity_rate = initial_sparsity_rate
        self.sparsity_rate = initial_sparsity_rate

    def build(self, input_shape):
        # Initialize the sparsity rate as a trainable variable
        self.sparsity_rate = self.add_weight(
            name='sparsity_rate',
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_sparsity_rate),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0, 1)  # Ensures sparsity rate is between 0 and 1
        )

    def call(self, inputs):
        # Calculate t_r based on the trainable sparsity rate
        sorted_inputs = tf.sort(inputs, direction='DESCENDING')
        num_features = tf.shape(inputs)[-1]
        index = tf.cast(self.sparsity_rate * tf.cast(num_features, tf.float32), tf.int32)

        # Create indices for gathering t_r_values for each example in the batch
        batch_range = tf.range(tf.shape(inputs)[0])
        indices = tf.stack([batch_range, tf.fill(tf.shape(batch_range), index)], axis=1)
        t_r_values = -tf.gather_nd(sorted_inputs, indices) + tf.reduce_max(inputs, axis=-1)

        # Calculate t-softmax
        max_inputs = tf.reduce_max(inputs, axis=-1)
        w_t = tf.nn.relu(inputs + tf.expand_dims(t_r_values, -1) - tf.expand_dims(max_inputs, -1))
        weighted_exp = tf.multiply(w_t, tf.exp(inputs))
        sum_weighted_exp = tf.reduce_sum(weighted_exp, axis=-1, keepdims=True)
        return weighted_exp / sum_weighted_exp
