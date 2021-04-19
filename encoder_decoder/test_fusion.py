import os

from skimage.transform import resize
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.applications.resnet import ResNet50

from config import config
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
from skimage import io

from encoder_decoder.data_gen import data_generator, DataHelperRMS_FUSION
from encoder_decoder.model_fusion import build_fusion_model, build_fusion_model_2

disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')

folder_path = config.test_dir + "/test"
ims = [x for x in os.listdir(folder_path) if x.endswith(".jpg")]
inception = MobileNetV2(weights='imagenet', include_top=False)

model = build_fusion_model_2()
model.compile(optimizer='adam', loss='mse',
              metrics=["accuracy"]
              )
model.load_weights(config.model_min_loss_out_emb)


def print_results_and_evaluate():
    accs=[]
    for i, im in enumerate(ims):
        f = f"{folder_path}/{im}"
        read = io.imread(f)
        read1 = resize(read, (128, 128), anti_aliasing=True) * 255
        print(im)
        gray = rgb2gray(read)  # HxW
        grayscale = gray2rgb(gray)  # HxWx3 model expects 3 channels
        B = 1  # batchsize
        resized = np.zeros((B, 224, 224, 3), dtype=np.float32)
        resized[0, :, :, :] = resize(grayscale, (224, 224), mode='constant')
        resized = preprocess_input(resized)  # has to be preprocessed as keras api suggests
        emb = inception.predict(resized)
        emb = emb[0]  # 1,7,7,1280
        n_emb = emb[2:6, 2:6, :]  # 4,4,1280
        n_emb = n_emb.reshape((16, 16, -1))  # 16x16x80
        n_emb = n_emb[np.newaxis, :, :, :]
        X_batch = np.zeros((1, config.H, config.W, 1))
        LAB = rgb2lab(1.0 / 255 * read1)  # HxWx3
        X_batch[0, :, :, 0] = LAB[:, :, 0]  # HxW L channel 0 to 100
        Y_batch = LAB[:, :, 1:]/128  # HxWx2 AB channels
        y = model.predict([X_batch, n_emb])
        cur = np.zeros((config.H, config.W, 3))
        cur[:, :, 0] = LAB[:, :, 0]
        cur[:, :, 1:] = y * 128
        io.imsave(f"{config.results_dir_emd}/{config.c}_00{i}_truth.jpg", read)
        io.imsave(f"{config.results_dir_emd}/{config.c}_00{i}_model_out.jpg", lab2rgb(cur))

        Y_batch = Y_batch[np.newaxis, :, :, :]
        loss, acc = model.evaluate([X_batch, n_emb], Y_batch)
        print(f"loss={loss}, mse accuracy={acc}")
        accs.append(acc)
    print(f"\nFinal average mse accuracy = {sum(accs)/len(accs)}")

helper = DataHelperRMS_FUSION()
print_results_and_evaluate()
