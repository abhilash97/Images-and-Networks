#detecting edges using manual convolutions
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color

import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\CNN\Images and Networks\Edge detection\Images\test2.jpg')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

from scipy import ndimage

k1 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])#for horizontal edges
k2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])#for vertical edges

x = ndimage.convolve(gray, k1, mode='constant', cval=0.0)
plt.imshow(x,cmap = plt.get_cmap('gray'))


import scipy.signal

edges = scipy.signal.convolve2d(gray, kernel, 'valid')
# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.09)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()