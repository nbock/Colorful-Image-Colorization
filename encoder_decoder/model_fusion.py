from tensorflow.keras.models import Sequential
import keras.callbacks
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer, Input, RepeatVector, Reshape, concatenate

from tensorflow.python.keras.models import Model

from config import config


def build_fusion_model():
    image_input = Input(shape=(config.H, config.W, 1))
    resnet_features_input = Input(shape=(1000,))
    kernel = 3

    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True)(image_input)
    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 64,64,32

    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 32,32,64

    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 16,16,128
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    # b,16,16,256

    # FUSION
    fusion_out = RepeatVector(16 * 16)(resnet_features_input)  # replicate 1000 features 16x16 times
    fusion_out = Reshape(([16, 16, 1000]))(fusion_out)  # b,16,16,1000
    fusion_out = concatenate([x, fusion_out], axis=3)  # concatenate to get b,16,16,1256
    fusion_out = Conv2D(256, (1, 1), activation='relu', padding='same', use_bias=True)(fusion_out)  # b,16,16,256
    # now depth shrinked from 1256 to 256 and shape is b,16,16,256

    # Decoder part
    # 16,16,256
    dec = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True)(fusion_out)
    dec = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True)(dec)
    # 16,16,64
    dec = UpSampling2D(size=(2, 2))(dec)
    # 32,32,64
    dec = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True)(dec)
    dec = UpSampling2D(size=(2, 2))(dec)
    # 64,64,32
    dec = Conv2D(16, (kernel, kernel), activation='relu', padding='same', use_bias=True)(dec)
    # 64,64,16
    dec = Conv2D(2, (kernel, kernel), activation='tanh', padding='same', use_bias=True)(dec)
    # 64,64,2
    dec = UpSampling2D(size=(2, 2))(dec)
    # 128,128,2 range -1,1

    model = Model(inputs=[image_input, resnet_features_input], outputs=dec)
    model.summary()
    return model


def build_fusion_model_2():
    image_input = Input(shape=(config.H, config.W, 1))
    resnet_features_input = Input(shape=(16, 16, 80,))
    kernel = 3

    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True)(image_input)
    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 64,64,32

    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 32,32,64

    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 16,16,128
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    # b,16,16,256

    # FUSION

    fusion_out = concatenate([x, resnet_features_input, resnet_features_input],
                             axis=3)  # concatenate to get b,16,16,336
    fusion_out = Conv2D(256, (1, 1), activation='relu', padding='same', use_bias=True)(fusion_out)  # b,16,16,256
    # now depth shrinked from 336 to 256 and shape is b,16,16,256

    # Decoder part
    # 16,16,256
    dec = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True)(fusion_out)
    dec = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True)(dec)
    # 16,16,64
    dec = UpSampling2D(size=(2, 2))(dec)
    # 32,32,64
    dec = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True)(dec)
    dec = UpSampling2D(size=(2, 2))(dec)
    # 64,64,32
    dec = Conv2D(16, (kernel, kernel), activation='relu', padding='same', use_bias=True)(dec)
    # 64,64,16
    dec = Conv2D(2, (kernel, kernel), activation='tanh', padding='same', use_bias=True)(dec)
    # 64,64,2
    dec = UpSampling2D(size=(2, 2))(dec)
    # 128,128,2 range -1,1

    model = Model(inputs=[image_input, resnet_features_input], outputs=dec)
    model.summary()
    return model


if __name__ == '__main__':
    m = build_fusion_model_2()
