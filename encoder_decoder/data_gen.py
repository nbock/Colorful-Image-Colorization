from keras_preprocessing.image import ImageDataGenerator
from skimage import io
from skimage.color import rgb2lab, gray2rgb, rgb2gray
from skimage.transform import rescale, resize
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

from config import config
from data_generator import DataHelper
import numpy as np

disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')

class DataHelperRMS_FUSION:
    def __init__(self, directory_path=config.train_dir):
        '''
        This is experimental for now because embedding features of transformed image may not be same
        '''
        self.datagen = ImageDataGenerator(
            # shear_range=0.2,
            # zoom_range=0.2,
            # rotation_range=20,
            # horizontal_flip=True,
            # validation_split=0.1
        )
        print("constructor called")

        # iterator of images in the directory
        self.train_iter = self.datagen.flow_from_directory(directory_path, target_size=(config.H, config.W),
                                                           batch_size=config.batch_size, class_mode=None, shuffle=False,
                                                           subset='training')
        self.validation_iter = self.datagen.flow_from_directory(directory_path, target_size=(config.H, config.W),
                                                                batch_size=config.batch_size, class_mode=None,
                                                                shuffle=False,
                                                                subset='validation')
        self.test_iter = self.datagen.flow_from_directory(config.test_dir, target_size=(config.H, config.W),
                                                          batch_size=1, class_mode=None,
                                                          shuffle=False)


def data_generator(helper, type="train"):
    iter = helper.train_iter
    if type == "val":
        iter = helper.validation_iter
    elif type == "test":
        iter = helper.test_iter
    iter.shuffle = False  # very important for filenames to work
    for batch_ in iter:
        b = batch_.shape[0]  # current size <=50
        B = iter.batch_size  # always 50 constant
        idx = (iter.batch_index - 1) * B

        # b_emb = np.zeros((b, 1000))
        b_emb = np.zeros((b, 16, 16, 80))
        if iter.batch_index:
            for i in range(b):
                im = iter.filenames[idx + i].split("/")[1]
                emb = np.load(f"{config.embedding_path}/{im}.npy")[0]
                b_emb[i, :] = emb
                # if iter.batch_index == 39:  #test to see if images file names are correct
                #     io.imsave(im, batch_[i, :, :, :])
        batch = 1.0 / 255 * batch_  # rgb2lab needs rgb in 0..1 range
        LAB_batch = rgb2lab(batch)  # BxHxWx3
        X_batch = LAB_batch[:, :, :, 0]  # BxHxW L channel 0 to 100
        X_batch = X_batch[:, :, :, np.newaxis]  # BxHxWx1
        Y_batch = LAB_batch[:, :, :, 1:]  # BxHxWx2 AB channels

        yield [X_batch, b_emb], Y_batch / 128



if __name__ == '__main__':
    helper = DataHelperRMS_FUSION()