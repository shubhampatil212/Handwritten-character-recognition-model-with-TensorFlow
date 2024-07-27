from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block

def create_handwriting_recognition_model(input_shape, output_classes, activation="leaky_relu", dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape, name="input_layer")
    normalized_inputs = layers.Lambda(lambda x: x / 255)(inputs)

    x = residual_block(normalized_inputs, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout_rate)
    x = residual_block(x, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout_rate)
    x = residual_block(x, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout_rate)

    x = residual_block(x, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout_rate)
    x = residual_block(x, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout_rate)

    x = residual_block(x, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout_rate)
    x = residual_block(x, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout_rate)
    x = residual_block(x, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout_rate)
    x = residual_block(x, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout_rate)

    feature_map = layers.Reshape((x.shape[-3] * x.shape[-2], x.shape[-1]))(x)

    lstm_output = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(feature_map)
    lstm_output = layers.Dropout(dropout_rate)(lstm_output)

    output = layers.Dense(output_classes + 1, activation="softmax", name="output_layer")(lstm_output)

    model = Model(inputs=inputs, outputs=output)
    return model