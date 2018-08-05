#visualizing the image within the covnet layers
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from convolveUsingFilters import img_read

img = img_read(1)
s = img.shape# 2592,1920,3

#creating a single convolutional layer
inputs = Input(shape=s)
# a layer instance is callable on a tensor, and returns a tensor
x = Conv2D(3, kernel_size = (3,3), strides = (1,1))(inputs)
model = Model(inputs=inputs, outputs=x)

#prediction in keras require a set of images/more than 1 image
#solution - add a dimension to the image array
ximg = np.expand_dims(img,axis=0)

#Now we can do the prediction
convolved_img = model.predict(ximg)

#visualizing what the convolutional layer sees
c1_img = np.squeeze(convolved_img,axis=0)
plt.imshow(c1_img)
plt.axis('off')
plt.savefig('C1 output with 3 filters.pdf')