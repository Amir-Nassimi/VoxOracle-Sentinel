
from singleton_decorator import singleton

from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Softmax, GlobalAveragePooling2D

from r_softmax import RSoftmax
from attention import NormalAttentionLayer, GAAPAttentionLayer


@singleton
class ModelBuilder:
    def __init__(self, input_shape):
        self.base_model = DenseNet121(weights='imagenet', include_top=False, pooling=None, input_shape=input_shape)

    def build_model(self, no_class, attention_type=None, softmax_type=None, initial_sparsity_rate=0.5):
        x = self.base_model.output

        if attention_type == 'normal':
            x = GlobalAveragePooling2D()(x)
            x = NormalAttentionLayer()(x)
        elif attention_type == 'gaap':
            x = GAAPAttentionLayer(self.base_model.layers[-1].output_shape[-1])(x)
            x = GlobalAveragePooling2D()(x)
        else:
            x = GlobalAveragePooling2D()(x)

        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(no_class)(x)

        if softmax_type == 'r_softmax':
            preds = RSoftmax(initial_sparsity_rate)(preds)
        else:
            preds = Softmax()(preds)

        model = Model(inputs=self.base_model.input, outputs=preds)
        return model
