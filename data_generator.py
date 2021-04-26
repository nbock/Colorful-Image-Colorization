# %%
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize

from config import config
import sklearn.neighbors as nn
import os


class DataHelper:
    def __init__(self, directory_path=config.train_dir):
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
                                                           batch_size=config.batch_size, class_mode=None, shuffle=True,
                                                           subset='training')
        self.validation_iter = self.datagen.flow_from_directory(directory_path, target_size=(config.H, config.W),
                                                                batch_size=config.batch_size, class_mode=None,
                                                                shuffle=True,
                                                                subset='validation')
        self.test_iter = self.datagen.flow_from_directory(config.test_dir, target_size=(config.H, config.W),
                                                          batch_size=1, class_mode=None,
                                                          shuffle=False)

        self.quantized_ab = np.load(os.path.join(config.color_data_path, "pts_in_hull.npy"))  # 313x2 (ab values)
        self.prior_factor = np.load(config.prior_factor_file_path)

        self.Q = self.quantized_ab.shape[0]
        self.nn_finder = nn.NearestNeighbors(n_neighbors=config.neighbours_soft_encoding, algorithm='ball_tree').fit(
            self.quantized_ab)

    def soft_Encoder(self, ab_batch):
        '''
        convert #Bxhxwx2 into Bxhxwx313(gaussian scores)
        '''
        B, h, w, _ = ab_batch.shape
        if B > config.batch_size:
            raise ValueError("invaid size")

        ab_batch_queries = ab_batch.reshape((-1, 2))  # Bhwx2
        distances, class_indices = self.nn_finder.kneighbors(ab_batch_queries)  # distances,indices =(BHWx5,BHWx5)
        # smoothing distances
        sigma = 5
        weights = np.exp(-distances * distances / (2 * sigma * sigma))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]  # Bhwx5 rows   divide by sum of 5 weights in column
        Y = np.zeros((B * h * w, self.Q), dtype=np.float32)  # Bhwx313 of 0s
        ix = np.arange(Y.shape[0]).reshape(-1, 1)

        # Now we want Y[i][class_indices[i]]=distances[i] for all i
        Y[ix, class_indices] = weights

        return Y.reshape(B, h, w, -1)  # Bxhxwx313 [5 probs per pixel others 0s]

        # check if g is same as Y
        # block = g[4, :, :, :]
        # (np.ravel(Y[4 * h * w :5 * h * w,:])==np.ravel(block)).all() must be true


def data_generator(helper, type="train"):
    iter = helper.train_iter
    if type == "val":
        iter = helper.validation_iter
    elif type == "test":
        iter = helper.test_iter

    for batch_ in iter:
        batch = 1.0 / 255 * batch_  # rgb2lab needs rgb in 0..1 range
        B = batch_.shape[0]
        lab_batch = rgb2lab(batch)  # BxHxWx3
        X_batch = lab_batch[:, :, :, 0]  # BxHxW L channel 0 to 100
        # X_batch = batch[:,:,:,0] #for grayscale as input instead of L channel
        lab_batch_resized = np.zeros((B, config.h, config.w, 3))  # Bxhxwx3
        for i in range(B):
            lab_batch_resized[i, :, :, :] = rescale(lab_batch[i, :, :, :], config.scale, anti_aliasing=True,
                                                    multichannel=True)
            # io.imsave("test"+str(i)+".png", lab2rgb(lab_batch_resized[i,:,:,:]))
        ab_batch = lab_batch_resized[:, :, :, 1:]  # Bxhxwx2
        # print("a channel",ab_batch[:,:,:,0].min(),ab_batch[:,:,:,0].max(),"b channel",ab_batch[:,:,:,1].min(),ab_batch[:,:,:,1].max())
        Y_batch = helper.soft_Encoder(ab_batch)  # Bxhxwx313
        X_batch = X_batch[:, :, :, np.newaxis]
        yield X_batch, Y_batch
