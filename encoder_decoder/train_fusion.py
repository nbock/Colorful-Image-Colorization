import datetime

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

from encoder_decoder.data_gen import DataHelperRMS_FUSION, train_generator
from encoder_decoder.model_fusion import build_fusion_model, build_fusion_model_2
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import keras.backend as K
from config import config
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_checkpoint = ModelCheckpoint(config.model_min_loss_out_emb, monitor='loss', verbose=1, save_best_only=True)
    model_checkpoint2 = ModelCheckpoint(config.model_min_val_loss_out_emd, monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,min_delta=0.001)

    new_model = build_fusion_model_2()
    new_model.compile(optimizer='adam',
                      loss='mse'
                      )


    callbacks = [es, model_checkpoint, model_checkpoint2]
    helper = DataHelperRMS_FUSION()

    '''
       model load or not setting
    '''
    # new_model.load_weights(config.model_min_loss_out_emb)
    new_model.fit(train_generator(helper),
                  steps_per_epoch=config.train_size // config.batch_size,
                  validation_data=train_generator(helper, train=False),
                  validation_steps=config.validation_size // config.batch_size,
                  epochs=config.epochs,
                  verbose=1,
                  callbacks=callbacks,
                  use_multiprocessing=False
                  )

