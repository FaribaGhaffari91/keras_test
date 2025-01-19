import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Reading dataset
red_wine = pd.read_csv("winequality-red.csv")

# What the dataset contains
print(red_wine.head())
print(red_wine.shape)

#Simple model with one neuron
model = keras.Sequential([
    layers.Dense(units = 1, input_shape = [1])
])

# weights of untrained model
w, b = model.weights

print("Weights\n{}\n\nBias\n{}".format(w, b))

#Create fake input
x = tf.linspace(1, 10, 10)
#output of single neuron (f(x) = w*x + b)
y = model.predict(x)

# To print our sample test
print("input (X) is:{}".format(x)) 
print("output (Y) is:{}".format(y)) # Note that y[i] = w[i] * x[i] + b


