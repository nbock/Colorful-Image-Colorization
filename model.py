from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from config import config
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.models import Model


def build_simple_model():
    scale = config.H / 256
    model = Sequential()
    model.add(InputLayer(input_shape=(config.H, config.W, 1)))
    # now 256x256x1
    model.add(Conv2D(int(64 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    model.add(Conv2D(int(64 * scale), (3, 3), activation='relu', padding='same', strides=2, use_bias=True))
    # now 128x128x64

    model.add(Conv2D(int(128 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    model.add(Conv2D(int(128 * scale), (3, 3), activation='relu', padding='same', strides=2, use_bias=True))
    # model.add(BatchNormalization())
    # now 64x64x128
    model.add(Conv2D(int(256 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    model.add(Conv2D(int(256 * scale), (3, 3), activation='relu', padding='same', strides=2))
    # model.add(BatchNormalization())
    # now 32x32x256
    model.add(Conv2D(int(512 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    # now 32x32x512
    model.add(Conv2D(int(256 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    # now 32x32x256
    # model.add(Conv2D(int(128 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    # model.add(BatchNormalization())

    model.add(UpSampling2D((2, 2)))
    # now 64x64x256

    outputs = Conv2D(config.Q, (1, 1), activation='softmax', padding='same', name='pred')
    model.add(outputs)
    # now 64x64x313  [0-1]

    # model.add(Conv2D(int(64 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(int(32 * scale), (3, 3), activation='relu', padding='same', use_bias=True))
    # model.add(Conv2D(2, (3, 3), activation='tanh', padding='same', use_bias=True))  # tanh
    # model.add(UpSampling2D((2, 2)))
    model.summary()
    return model


kernel = 3


def build_zhangs_model():
    input_tensor = Input(shape=(config.H, config.W, 1))
    # x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal")(input_tensor)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal",
               strides=(2, 2))(input_tensor)
    # x = BatchNormalization()(x)

    # x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal",
    #            kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', kernel_initializer="he_normal",
               strides=(2, 2))(x)
    # x = BatchNormalization()(x)

    # x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_3', kernel_initializer="he_normal",
               strides=(2, 2))(x)
    # x = BatchNormalization()(x)

    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_1',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_2',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_3',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = BatchNormalization()(x)

    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_1',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_2',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_3',
               kernel_initializer="he_normal")(x)
    # x = BatchNormalization()(x)

    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_1',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_2',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_3',
               kernel_initializer="he_normal")(x)
    # x = BatchNormalization()(x)

    # x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_1',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_2',
               kernel_initializer="he_normal")(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_3',
               kernel_initializer="he_normal")(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3',
    #            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    # x = BatchNormalization()(x)

    outputs = Conv2D(config.Q, (1, 1), activation='softmax', padding='same', name='pred')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    model.summary()
    return model


def build_zhangs_model_2():
    input_tensor = Input(shape=(config.H, config.W, 1))
    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True)(input_tensor)
    x = Conv2D(32, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 64,64,32

    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 32,32,64

    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', use_bias=True, strides=2)(x)
    # 16,16,128

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    # 16,16,256

    x = UpSampling2D(size=(2, 2))(x)
    # 32,32,256
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', use_bias=True)(x)
    # 32,32,256

    # x = BatchNormalization()(x)

    outputs = Conv2D(config.Q, (1, 1), activation='softmax', padding='same', name='pred')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    model.summary()
    return model


if __name__ == '__main__':
    m = build_zhangs_model_2()
