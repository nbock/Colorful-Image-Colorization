import os

from skimage.transform import resize
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.applications.resnet import ResNet50

from config import config
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
from skimage import io

disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')

from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#
folder_path = config.train_dir + "/train"
ims = [x for x in os.listdir(folder_path) if x.endswith(".jpg")]
import tensorflow as tf

inception = MobileNetV2(weights='imagenet', include_top=False)
# inception = ResNet50(weights='imagenet', include_top=False)
inception.graph = tf.compat.v1.get_default_graph()

for im in ims:
    f = f"{folder_path}/{im}"
    read = io.imread(f)
    print(im)
    gray = rgb2gray(read)  # HxW
    grayscale = gray2rgb(gray)  # HxWx3 model expects 3 channels
    B = 1  # batchsize
    resized = np.zeros((B, 224, 224, 3), dtype=np.float32)
    resized[0, :, :, :] = resize(grayscale, (224, 224), mode='constant')
    resized = preprocess_input(resized)  # has to be preprocessed as keras api suggests
    emb = inception.predict(resized)  # 1,7,7,1280
    emb = emb[0]  # 7,7,1280
    n_emb = emb[2:6, 2:6, :]  # 4,4,1280
    n_emb = n_emb.reshape((16, 16, -1))   #16,16,80
    np.save(f"{config.embedding_path}/{im}.npy", n_emb)

print("")
