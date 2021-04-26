import datetime

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_gen import DataHelperRMS_FUSION, data_generator
from model_fusion import build_fusion_model, build_fusion_model_2
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import keras.backend as K
from config import config
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_checkpoint = ModelCheckpoint(config.model_min_loss_out_emb, monitor='loss', verbose=1, save_best_only=True)
    model_checkpoint2 = ModelCheckpoint(config.model_min_val_loss_out_emd, monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7,min_delta=0.001)

    new_model = build_fusion_model_2()
    new_model.compile(optimizer='adam',
                      loss='mse'
                      )


    callbacks = [es, model_checkpoint, model_checkpoint2]
    helper = DataHelperRMS_FUSION()

    '''
       model load or not setting
    '''
    new_model.load_weights(config.model_min_loss_out_emb)
    new_model.fit(data_generator(helper),
                  steps_per_epoch=config.train_size // config.batch_size,
                  validation_data=data_generator(helper, type="val"),
                  validation_steps=config.validation_size // config.batch_size,
                  epochs=config.epochs,
                  verbose=1,
                  callbacks=callbacks,
                  use_multiprocessing=False
                  )

