#OCR using MNIST
#a convolutional neural network
#import os
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import struct
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# Functions to load MNIST images and unpack into train and test set.
# - loadData reads image data and formats into a 28x28 long array
# - loadLabels reads the corresponding labels data, 1 for each image
# - load packs the downloaded image and labels data into a combined format to be read later by 
#   CNTK text reader 
train_samples = 60000 #number of training samples
test_samples = 10000 #number of test samples
def loadData(path,img_n):
    
    with gzip.open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))# extracting the first 4 bytes and converting into asn integer
        #since stored in big endian format.need to flick it, or convert it.data like in C array
        # Read magic number.
        if n[0] != 0x3080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', f.read(4))[0]
        if n != img_n:
            raise Exception('Invalid file: expected {0} entries.'.format(img_n))
        crow = struct.unpack('>I', f.read(4))[0]
        ccol = struct.unpack('>I', f.read(4))[0]
        if crow != 28 or ccol != 28:
            raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
        res = np.fromstring(f.read(img_n * crow * ccol), dtype = np.uint8)
            #A new 1-D array initialized from text data in a string.
        #os.remove(path)
    return res.reshape((img_n, crow * ccol))

def loadLabels(path,img_n):

    with gzip.open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))
        # Read magic number.
        if n[0] != 0x1080000:
            raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
        n = struct.unpack('>I', f.read(4))
        if n[0] != img_n:
            raise Exception('Invalid file: expected {0} rows.'.format(img_n))
            # Read labels.
        res = np.fromstring(f.read(img_n), dtype = np.uint8)
    
        #os.remove(path)
    return res.reshape((img_n, 1))

def load_dataset(path_data, path_labels, img_n):
    data = loadData(path_data,img_n)
    labels = loadLabels(path_labels,img_n)
    return np.hstack((data, labels))#to merge them horizontally into a single array of data and labels


train_data = load_dataset('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',train_samples)#includes both the features and labels
test_data = load_dataset('t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz',test_samples)#includes both the features and labels
"""#visualizing data#

sample_number = 5004
plt.imshow(train_data[sample_number,1:].reshape(28,28), cmap="gray_r")
plt.axis('off')#for reshaping into x*x, the array length shouldbe x*x
print("Image Label: ", train_data[sample_number,-1])
"""
def build_model():
    model = Sequential()
    model.add(Conv2D(8, (5, 5),strides = (2,2), padding='same',data_format="channels_last", input_shape=(28,28), activation='relu'))
    model.add(Conv2D(16, (5, 5), strides = (2,2), padding = 'same', activation='relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model
    
def labelEncode(labels):
    #one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
    lab_encoder = LabelEncoder()
    int_encoder = lab_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoder = int_encoder.reshape(len(int_encoder), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoder)
    return onehot_encoded

input_dim = 784
a_test = []
a_train = []
num_output_classes = 10
labels_train = train_data[:,-1]
labels_test = test_data[:,-1]
x1 = np.array([28,28])
x2 = np.array([28,28])
features_train = (train_data[:,:len(train_data[0])-1])#scaling
for j in range(0,len(features_train)):
    x1 = train_data[j,:-1].reshape(28,28)
    a_train.append(x1)
features_test = (test_data[:,:len(test_data[0])-1])
for i in range(0,len(features_test)):
    x2 = test_data[i,:-1].reshape(28,28)
    a_test.append(x2)

labels_train = labelEncode(labels_train)
labels_test = labelEncode(labels_test)
model = build_model()
sgd = optimizers.SGD(lr=0.2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(a_train, labels_train, epochs=20, batch_size=32)
print('Done training')
score = model.evaluate(a_test, labels_test, batch_size=32)
print(score)


predictions = model.predict(features_test)

pred = [np.argmax(predictions[i]) for i in range(len(predictions))]
gtlabel = [np.argmax(labels_test[i]) for i in range(len(labels_test))]
print("Image Labels: ", gtlabel[:25])
print("predicted:    ", pred[:25])

sample_number = 5
img_pred = pred[sample_number]
plt.imshow(features_test[sample_number].reshape(28,28), cmap="gray_r")
plt.axis('off')
print('Number: ',img_pred)
