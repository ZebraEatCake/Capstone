from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def VGGupdated(input_tensor=None,classes=3):    
   
    img_rows, img_cols = 244, 244   # img size
    img_channels = 3 # rgb

    img_dim = (img_rows, img_cols, img_channels)
   
    img_input = Input(shape=img_dim)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
   
     
    model = Model(inputs = img_input, outputs = x, name='VGGdemo')


    return model

model = VGGupdated(classes = 3) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

dataset_path = os.listdir('Z:\Sunway\s9\Capstone\Dataset')

plant_types = os.listdir('Z:\Sunway\s9\Capstone\Dataset')
print (plant_types)  #print number of classes 

plants = []
for item in plant_types:
 # Get all the file names
 all_plants = os.listdir('Z:\Sunway\s9\Capstone\Dataset' + '/' +item)
 #print(all_shoes)

 # Add them to the list
 for plant in all_plants:
    plants.append((item, str('Z:\Sunway\s9\Capstone\Dataset' + '/' +item) + '/' + plant))
    #print(plants)

# Build a dataframe        
plants_df = pd.DataFrame(data=plants, columns=['plant type', 'image'])
print(plants_df.head())
#print(plants_df.tail())

# Let's check how many samples for each category are present
print("Total number of plants in the dataset: ", len(plants_df))

plant_count = plants_df['plant type'].value_counts()

print("plants in each category: ")
print(plant_count)

import cv2
path = 'Z:\Sunway\s9\Capstone\Dataset/'


im_size = 244

images = [] #to store images after resizing
labels = []

for i in plant_types:
    data_path = path + str(i)  
    filenames = [i for i in os.listdir(data_path) ]
   
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

images = np.array(images) # convert image into array for model

images = images.astype('float32') / 255.0
print(images.shape)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Step 1: Get the plant type column
y = plants_df['plant type'].values

# Step 2: Label encode
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 3: Reshape for one-hot
y = y.reshape(-1, 1)

# Step 4: One-hot encode
onehot_encoder = OneHotEncoder(sparse_output=False) 
Y = onehot_encoder.fit_transform(y)

print(Y.shape)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


images, Y = shuffle(images, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

#inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model.fit(train_x, train_y, epochs = 100, batch_size = 32)  
preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))