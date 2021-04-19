from skimage.color import rgb2lab, lab2rgb
from skimage.transform import rescale
from tensorflow.python.framework.ops import disable_eager_execution

from config import config
from data_generator import DataHelper, data_generator
import numpy as np
# def get_colored_image(Limage):
from main import categorical_mine
from model import build_zhangs_model, build_zhangs_model_2
from skimage import io
import tensorflow as tf
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')

class Tester:
    def __init__(self, path=config.model_min_loss_out):
        self.helper = DataHelper()
        self.n = len(self.helper.test_iter.filenames)
        self.model = build_zhangs_model_2()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           # loss=categorical_mine,
                           metrics=["accuracy", tf.keras.metrics.CategoricalCrossentropy()]
                           )
        self.model.load_weights(path)
        self.a = self.helper.quantized_ab[:, 0]  # 313
        self.b = self.helper.quantized_ab[:, 1]  # 313

    def lchannel_to_color(self, L):
        L_in = L[np.newaxis, :, :, np.newaxis]  # 1,H,W,1
        ab_q = self.model.predict(L_in)[0]  # h,w,313

        # annealed mean along class axis
        ab_q_exp = np.exp(np.log(ab_q) / config.T)  # h,w,313
        sums = np.sum(ab_q_exp, axis=2)  # h,w
        sums = sums[:, :, np.newaxis]  # h,w,1
        ab_q_annealed = ab_q_exp / sums  # h,w,313
        a = np.sum(ab_q_annealed * self.a, axis=2)  # h,w
        b = np.sum(ab_q_annealed * self.b, axis=2)  # h,w
        a = rescale(a, config.upscale, anti_aliasing=True, multichannel=False)
        b = rescale(b, config.upscale, anti_aliasing=True, multichannel=False)

        LAB = np.zeros((config.H, config.W, 3), dtype=np.float32)
        LAB[:, :, 0] = L
        LAB[:, :, 1] = a
        LAB[:, :, 2] = b
        RGB = lab2rgb(LAB)
        return RGB

    def print_results(self):

        for i in range(self.n):
            RGB = self.helper.test_iter.next()[0]  # H,W,3
            RGB = 1.0 / 255 * RGB  # rgb2lab needs rgb in 0..1 range
            LAB = rgb2lab(RGB)  # H,W,3
            L = LAB[:, :, 0]  # H,W
            RGB_out = self.lchannel_to_color(L)
            io.imsave(f"{config.results_dir}/{config.c}_00{i}_truth.jpg", RGB)
            io.imsave(f"{config.results_dir}/{config.c}_00{i}_model_out.jpg", RGB_out)

    def evaluate(self):
        test_generator = data_generator(self.helper, type="test")
        self.helper.test_iter.batch_size = self.n
        for X_batch, Y_batch in test_generator:
            loss, acc, ac2 = self.model.evaluate(X_batch, Y_batch)
            print(f"classification accuracy={acc}, cross entropy={ac2}")
            break


if __name__ == '__main__':
    t = Tester(path=config.model_min_loss_out)
    # t.print_results()
    t.evaluate()
