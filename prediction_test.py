#!/home/chieh/bin/anaconda3/envs/py3.9t2.9/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from tensorflow import keras
classifier = keras.models.load_model("classifier.h5")


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
	'Dataset/training_set',
	target_size=(64,64),
	batch_size=32,
	class_mode='binary')

'''
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
	'Dataset/test_set',
	target_size=(64,64),
	batch_size=32,
	class_mode='binary')
'''

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('negative.jpg', target_size = (64, 64))
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

