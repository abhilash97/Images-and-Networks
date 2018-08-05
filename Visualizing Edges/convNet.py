#visualizing the image within the covnet layers
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model

import matplotlib.image as mpimg
from convolveUsingFilters import img_read

img = img_read(1)
s = img.shape# 2592,1920,3
inputs = Input(shape=s)

# a layer instance is callable on a tensor, and returns a tensor
x = Conv2D(8, kernel_size = (2,2), strides = (1,1))(inputs)

model = Model(inputs=inputs, outputs=x)
model.predict()
print(type(x))
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(data, labels)
