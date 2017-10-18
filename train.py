import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from dataloader import dataloader
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os.path


# Load Data
data = dataloader(train_path = "./training_alphabet/", test_path="./test_alphabet/", data_type ="alphabets")
x_train, y_train = data.training_set()
y_train = keras.utils.to_categorical(y_train, num_classes=24)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
filepath="./weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

if os.path.isfile(filepath):
	model = load_model(filepath)
else:
	model = Sequential()
	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(24, activation='softmax'))

	optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

epochs = 200

x_test, y_test = data.test_set()
y_test = keras.utils.to_categorical(y_test, num_classes=24)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=30, epochs=epochs, validation_data = (x_test,y_test), callbacks=[tbCallBack,checkpoint])

