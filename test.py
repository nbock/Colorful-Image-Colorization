from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

from skimage import io

im=io.imread("/Users/naveen/Documents/ML local Data/butte/train/00000001.jpg")
i_=1.0 / 255*im
im2=rgb2lab(i_)
im3=rgb2lab(im)
io.imsave("f2.jpg",lab2rgb(im2))