#!/home/chieh/bin/anaconda3/envs/py3.9t2.9/bin/python

#Convolutional Neural network

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

#Importing the Keras libaries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#'''
#Initialize the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["binary_accuracy"])
#'''

#from tensorflow import keras
#classifier = keras.models.load_model("classifier.h5")


#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
	'Dataset/training_set',
	target_size=(64,64),
	batch_size=32,
	class_mode='binary')

test_set = test_datagen.flow_from_directory(
	'Dataset/test_set',
	target_size=(64,64),
	batch_size=32,
	class_mode='binary')




from IPython.display import display
from PIL import Image

#classifier.fit_generator(
classifier.fit(
	training_set.repeat(),
	steps_per_epoch=int(75/batch_size),
	epochs=10,
	validation_data=test_set.repeat(),
	validation_steps=int(75/batch_size))
classifier.save("classifier.h5")


#'''

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('random.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image /= 255.0
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
	prediction = 'dog'
else:
	prediction = 'cat'

print(training_set.class_indices, result[0][0], prediction)

#'''


