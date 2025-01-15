import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from learntools.core import binder
#binder.bind(globals())
#from learntools.deep_learning_intro.ex1 import *

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)

red_wine = pd.read_csv("winequality-red.csv")
print(red_wine.head())
print(red_wine.shape)

model = keras.Sequential([
    layers.Dense(units = 1, input_shape = [1])
])

w, b = model.weights

print("Weights\n{}\n\nBias\n{}".format(w, b))

#print(help(tf.linspace))
x = tf.linspace(-1.0, 10, 1)
y = model.predict(x)
print(y)