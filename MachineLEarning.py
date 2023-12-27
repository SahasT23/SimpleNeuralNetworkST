import os
import cv2  # Computer vision to process and load the images
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

mnist = tf.keras.datasets.mnist  # commonly used MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x = pixel data, y = classification (digits/data)
# Training data - values that we already know and use to aide the learning process
# Test data - values that the computer will read
x_train = tf.keras.utils.normalize(x_train, axis=1)  # Normalising is converting RGB values to lie in the range 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1) # Pre-processing

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu')) # relu = rectify linear unit
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # How likely an image is the correct digit (probability)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
