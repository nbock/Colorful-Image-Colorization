import datetime

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_simple_model, build_zhangs_model, build_zhangs_model_2
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import keras.backend as K
from data_generator import DataHelper, train_generator
from config import config
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')

prior_factor = np.load(config.prior_factor_file_path)
# this is important because the custom loss functions must work with keras tensors(not numpy arrays)
# and also of same data type
prior_factor = K.cast(prior_factor, dtype='float32')

def categorical_mine(y_true, y_pred):
    ''' This is where we add rarity weights to losses as per the research paper'''
    # ytrue= Bxhxwx313
    # ypred= Bxhxwx313 from model output

    # must return Bxhxw losses

    # cross_ent = tf.keras.losses.categorical_crossentropy(y_true,y_pred)  # Bxhxw losses per pixel
    cross_ent = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    idx_max = K.argmax(y_true, axis=-1)  # Bxhxw

    pixels_weights = K.gather(prior_factor, idx_max)
    ## wights same dim as idx_max

    # multiply cross_ent loss per pixel by weights
    cross_ent = cross_ent * pixels_weights
    # RARE PIXELS HAVE High Prior weights so THE LOSSES OF  RARE PIXELS ARE NOW MORE DOMINANT
    # cross_ent = K.mean(cross_ent, axis=-1)  # must return Nxhxw size losses
    # multiply cross_ent loss per pixel by weights
    # print("ytrue shape", y_true.shape, "y_pred shape", y_pred.shape, "cross ent shape", cross_ent.shape, "idx shape",
    #       idx_max.shape,
    #       "weights", weights.shape)

    '''Incompatible shapes: [16,32,313]weights vs. [16,32,32]crossent
    '''
    # K.reshape(cross_ent,(16,32,32))
    #todo remove these comments
    return cross_ent


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_checkpoint = ModelCheckpoint(config.model_min_loss_out, monitor='loss', verbose=1, save_best_only=True)
    model_checkpoint2 = ModelCheckpoint(config.model_min_val_loss_out, monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    new_model = build_zhangs_model_2()
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
    new_model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),


                      # loss=categorical_mine
                      )


    callbacks = [es, model_checkpoint, model_checkpoint2]
    helper = DataHelper()

    '''
    model load or not setting
    '''
    new_model.load_weights(config.model_min_loss_out)
    new_model.fit(train_generator(helper),
                  steps_per_epoch=config.train_size // config.batch_size,
                  validation_data=train_generator(helper, train=False),
                  validation_steps=config.validation_size // config.batch_size,
                  epochs=config.epochs,
                  verbose=1,
                  callbacks=callbacks,
                  use_multiprocessing=False
                  )

