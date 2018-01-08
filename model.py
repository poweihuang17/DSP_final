import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Activation, LeakyReLU
from keras.layers import Conv1D, Dropout, Dropout, Add, Concatenate

def create_model(max_steps, filters_list=[64, 128, 256, 256], kernel_size_list=[65, 33, 17, 9]):
    assert len(filters_list) == len(kernel_size_list)

    # encoder part
    input_layer = Input(shape=(max_steps, 1))
    x = input_layer

    encoder_layers = list()

    for filters, kernel_size in zip(filters_list, kernel_size_list):
        x = Conv1D(filters, kernel_size, padding='same') (x)
        x = Dropout(0.5) (x)
        x = LeakyReLU() (x)
        print(x)
        encoder_layers.append(x)

    # decoder part
    for shortcut, filters, kernel_size in zip(reversed(encoder_layers), reversed(filters_list), reversed(kernel_size_list)):
        x = Conv1D(filters * 2, kernel_size, padding='same') (x)
        x = Dropout(0.5) (x)
        x = LeakyReLU() (x)
        x = Concatenate() ([x, shortcut])
        print(x)

    # final conv layer
    x = Conv1D(1, 9, padding='same') (x)
    output_layer = Add() ([x, input_layer])
    print(x)

    model = Model(
        inputs=[input_layer],
        outputs=[output_layer]
    )
    return model
