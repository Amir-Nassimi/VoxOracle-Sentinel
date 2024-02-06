from tensorflow import matmul
from tensorflow.keras.layers import Layer, Conv1D
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.backend import tanh, dot, softmax
from tensorflow.keras.initializers import Constant as Constant_Initializer


class NormalAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(NormalAttentionLayer, self).__init__(**kwargs)
        self.W = None
        self.b = None

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(NormalAttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tanh(dot(x, self.W) + self.b)
        a = softmax(e, axis=1)
        output = x * a
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class GAAPAttentionLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(GAAPAttentionLayer, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # Convolutional layers for Q, K, V
        self.query_conv = Conv1D(filters=self.filters, kernel_size=1, strides=1, padding='same')
        self.key_conv = Conv1D(filters=self.filters, kernel_size=1, strides=1, padding='same')
        self.value_conv = Conv1D(filters=self.filters, kernel_size=1, strides=1, padding='same')
        self.attention_conv = Conv1D(filters=self.filters, kernel_size=1, strides=1, padding='same')

        # Learnable alpha parameter
        self.alpha = self.add_weight(name='alpha',
                                     shape=(1,),
                                     initializer=Constant_Initializer(0.5),
                                     trainable=True,
                                     constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0))

        super(GAAPAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Generate Q, K, V
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        # Compute attention scores
        scores = matmul(query, key, transpose_b=True)

        # Apply softmax to the scores
        attention_weights = softmax(scores, axis=-1)

        # Apply attention to value
        attention_output = matmul(attention_weights, value)

        # Apply the 1x1 conv layer to the attention output
        attention_feature_map = self.attention_conv(attention_output) * self.alpha

        attention_feature_map *= inputs

        return attention_feature_map
