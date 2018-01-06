import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Activation, LeakyReLU
from keras.layers import Conv1D, Dropout, Dropout, Add, Concatenate

def create_model(max_steps, filters_list=[128, 256, 512, 512, 512, 512, 512, 512], kernel_size_list=[65, 33, 17, 9, 9, 9, 9, 9]):
    assert len(filters_list) == len(kernel_size_list)

    input_layer = Input(shape=(max_steps, 1))
    x = input_layer

    # downsampling blocks
    downsampling_layers = list()

    for filters, kernel_size in zip(filters_list, kernel_size_list):
        x = Conv1D(filters, kernel_size, padding='same') (x)
        x = Dropout(0.5) (x)
        x = LeakyReLU() (x)
        downsampling_layers.append(x)

    # upsampling blocks
    for shortcut, filters, kernel_size in zip(reversed(downsampling_layers), reversed(filters_list), reversed(kernel_size_list)):
        x = Conv1D(filters * 2, kernel_size, padding='same') (x)
        x = Dropout(0.5) (x)
        x = LeakyReLU() (x)
        x = Concatenate() ([x, shortcut])

    # final conv layer
    x = Conv1D(1, 9, padding='same') (x)
    output_layer = Add() ([x, input_layer])

    model = Model(
        inputs=[input_layer],
        outputs=[output_layer]
    )
    return model
