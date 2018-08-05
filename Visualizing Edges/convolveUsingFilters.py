#visualizing edges using manual convolutions
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color

import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def img_read(no_gray):
    img = mpimg.imread(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\CNN\Images and Networks\Visualizing Edges\Images\test3.jpg')     
    if no_gray==0:
        gray = rgb2gray(img)
    else:
        gray = img        
    if __name__=="__main__":
        plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.colorbar()
        #plt.show()
        plt.savefig('grayscale.pdf')
    
    return gray


k1 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])#for horizontal edges
k2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])#for vertical edges

#using 1D ndimage convolve without any image sharpenning
"""from scipy import ndimage
x = ndimage.convolve(gray, k1, mode='constant', cval=0.0)
plt.imshow(x,cmap = plt.get_cmap('gray'))
"""
def main():
    import scipy.signal
    from skimage import exposure

    gray = img_read()
    edges = scipy.signal.convolve2d(gray, k2, 'valid')
    # Adjust the contrast of the filtered image by applying Histogram Equalization
    edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.09)
    plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
    plt.axis('off')
    #plt.show()
    plt.savefig('Vertical edge detection-clock_tower.pdf')

    #changing the filters into some random valued array
    #kx = np.array([[1,2,3],[2,3,4],[1,6,7]])
    #edgex = scipy.signal.convolve2d(gray, k1, 'valid')
    #edges_equalizedx = exposure.equalize_adapthist(edgex/np.max(np.abs(edgex)), clip_limit=0.09)
    #plt.imshow(edges_equalizedx, cmap=plt.cm.gray)    # plot the edges_clipped

if __name__=="__main__":
    main()
else:
    print("nothing to say")
    
