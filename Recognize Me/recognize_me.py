
# coding: utf-8

# In[2]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from PIL import Image


img = Image.open(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\Recognize\Images\t1.jpg')
#img = img.crop((1,1,50,100))
#img.save(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\Recog_me\tmod.jpg')
plt.imshow(img)
plt.axis('off')


# In[3]:

imgN = img.convert('LA')
plt.imshow(imgN)


# In[ ]:

#-------------Working with a single Image-------------


# In[5]:

dat = np.array(img.getdata())
datN = np.array(imgN.getdata())


# In[34]:

def convert_gscale(datN):
    data = np.empty([datN.shape[0]])
    for i in range(0,datN.shape[0]):
        data[i] = np.average(datN[i])
    #print(data.shape)
    return data


# In[58]:

print(dat.shape)
print(datN.shape)


# In[36]:

data = convert_gscale(datN)


# In[84]:

data.shape
data.reshape(1920,2560)


# In[42]:

#-----------------components of original image----------


r = dat[:,0] #r - component
g = dat[:,1] #g - component
b = dat[:,2] #b - component
print(r,g,b)
#Y_gray = 0.299*r + 0.587*g + 0.114*b
#print(Y_gray)
#Y_gray.shape


# In[9]:

r.shape, type(r)


# In[10]:

r = r.reshape(1920, 2560)
g = g.reshape(1920, 2560)
b = b.reshape(1920, 2560)
print(r.shape,g.shape,b.shape)
#---------------------------------------------


# In[12]:

dat = dat.reshape(1920,2560,3)
data = data.reshape(1920,2560,1)
print(dat.shape)
print(data.shape)


# In[ ]:

#-------------------Done! preprocessing a single image. Now for the whole dataset-----------------


# In[20]:

#--------load all the images--------------------
features = np.empty([1920,2560])
print(features.shape)
print(features)


# In[29]:

img = Image.open(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\Recognize\Images\t20.jpg')
img = img.convert('LA')
datx = np.array(img.getdata())
#print(datx, datx.shape)
datx = convert_gscale(dat1)
print(datx, datx.shape)
datx = dat1.reshape(1920,2560)


# In[26]:

print(datx, datx.shape)


# In[ ]:

for i in range(1,25):
    img = Image.open(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\Recognize\Images\t'+str(i)+'.jpg')
    img = img.convert('LA')
    dat1 = np.array(img.getdata())
    dat1 = convert_gscale(dat1)
    dat1 = dat1.reshape(1920,2560)
    features = np.append(features,dat1, axis=0)


# In[82]:

import pandas as pd

labels = pd.read_excel(r'C:\Users\user\Desktop\Abhilash\Imp\Deep_Learning\Neural Networks\Recognize\labels_train.xlsx')
print(labels,labels.shape)


# In[83]:

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def label_encode(labels):
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)
    return labels

#print(labels)
def one_hot_encode(labels):
    enc = OneHotEncoder(sparse = False)
    labels = labels.reshape(len(labels), 1)
    labels = enc.fit_transform(labels)#.toarray()
    return labels

labels = label_encode(labels)
print(labels)
labels = one_hot_encode(labels)
print(labels)


# In[19]:

# the cnn model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (6, 6), activation='relu', input_shape = (1920, 2560, 3)))
    
model.add(BatchNormalization())
    
model.add(Conv2D(filters = 16, kernel_size = (4, 4), activation='relu'))
model.add(BatchNormalization())
    
model.add(MaxPool2D(strides = (2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(filters = 16, kernel_size = (4, 4), activation='sigmoid'))
model.add(Conv2D(filters = 16, kernel_size = (4, 4), activation='relu'))    
model.add(BatchNormalization())    
model.add(MaxPool2D(strides = (2,2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.10))
model.add(Dense(1))
model.add(Activation('softmax'))

