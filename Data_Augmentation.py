#!/home/chieh/bin/anaconda3/envs/py3.9t2.9/bin/python

#Convolutional Neural network

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

#Importing the Keras libaries and packages
import glob
import shutil
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D
from keras.models import Sequential

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
	horizontal_flip=True,
	rotation_range=45,
	width_shift_range=.15,
	height_shift_range=.15
	)

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
history=classifier.fit(
		training_set,
		steps_per_epoch=167,
		epochs=10,
		validation_data=test_set,
		validation_steps=63)
classifier.save("classifier.h5")

#'''

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#model loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('figure/model loss.png')
plt.show()
#model accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('figure/model accuracy.png')
plt.show()

#confusion matrix



import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('positive.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image /= 255.0
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
	prediction = 'positive'
else:
	prediction = 'negative'

print(training_set.class_indices, result[0][0], prediction)

#'''