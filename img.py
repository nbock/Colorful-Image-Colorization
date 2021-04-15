from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import rescale

from config import config
from data_generator import DataHelper
import numpy as np
# def get_colored_image(Limage):
from model import build_zhangs_model, build_zhangs_model_2
from skimage import io


a=io.imread("gray.jpg")
a=rgb2gray(a)
io.imsave("gray.jpg",a)