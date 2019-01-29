# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:46:22 2019

@author: Qasim Arshad
"""

#Importing Keras libraires
from keras.models import Sequential
from keras.models import Convolution2D
from keras.models import MaxPooling2D
from keras.models import Flatten
from keras.models import Dense


# ---- PART + 1 ----

#Inilize the CNN
classifier = Sequential()

#Step 1 -- Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation = 'relu'))

#Step 2 -- Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 -- Flattening
classifier.add(Flatten())

#Step 4 -- Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Step 5 -- Compiling the CNN
classifier.compile(optimizer = 'adam',loss= 'binary_crossentropy',matrics = ['accuracy'])



# ---- PART + 2 ----

#step 6 -- Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontol_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_dictionary(
        'dataset/training_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode= 'binary')

test_set = test_datagen.flow_from_dictionary(
        'dataset/test_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode= 'binary')


from IPython.display import display
from PIL import Image
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data = test_set,
        validation_steps=800)

import numpy as np
from keras.processing import image
test_image = image.load_img('random.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0)
result= classifier.predict(test_image)
training_set.class_indices

if (result[0][0] >= 0.5):
    prediction ='Dog'
else:
    prediction ='Cat'

print(prediction)







