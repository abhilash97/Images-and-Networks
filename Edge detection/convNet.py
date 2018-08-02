#visualizing the image within the covnet layers
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model

import matplotlib.image as mpimg
from convolveUsingFilters import img_read

gray = img_read()

inputs = Input(shape=gray.shape)

# a layer instance is callable on a tensor, and returns a tensor
x = Conv2D(8, kernel_size = (2,2), strides = (1,1),activation = 'relu')(inputs)
x = MaxPooling2D(pool_size = (2,2), strides = None)(x)

#predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=x)
print(type(x))
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(data, labels)
