#
#Welcome to this Colab where you will train your first Machine Learning model!
#
#We'll try to keep things simple here, and only introduce basic concepts. Later Colabs will cover more advanced problems.
#
# The problem we will solve is to convert from Celsius to Fahrenheit, where the approximate formula is:
#
# 𝑓=𝑐×1.8+32
#
# Of course, it would be simple enough to create a conventional Python function that directly performs this calculation, but that wouldn't be machine learning.
#
# Instead, we will give TensorFlow some sample Celsius values (0, 8, 15, 22, 38) and their corresponding
# Fahrenheit values (32, 46, 59, 72, 100). Then, we will train a model that
# figures out the above formula through the training process.

### https://classroom.udacity.com/courses/ud187/lessons/e0c70c77-5584-4f83-a47b-a67a6172ae75/concepts/ac6c6991-8096-4c7a-bad1-706f7e3d36f1


import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#Use the model to predict values
print(model.predict([100.0]))

#Looking at the layer weights
print("These are the layer variables: {}".format(l0.get_weights()))

#A little experiment
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))


