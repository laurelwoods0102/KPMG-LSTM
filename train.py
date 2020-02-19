import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

from preprocess import Preprocess
from plotting import show_plot

# Settings
with open("settings.json") as j:
    settings = json.load(j)

if settings["SEED"]:
    tf.random.set_seed(settings["SEED"])


# Dataset
dataframe = pd.read_csv("./data/lalavla-강남구-Nail.csv")
#dataframe = dataframe[:576]

data = dataframe["qty"]
data.index = dataframe["date"]

preprocess = Preprocess(data)
mean, std = preprocess.normalization()
train_dataset, val_dataset, shape = preprocess.create_dataset()

normalize = np.array([mean, std])
np.savetxt('normalize.csv', normalize, delimiter=", ")

# Model
model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, input_shape=shape[1:]),     # Pylint error
        tf.keras.layers.Dense(1)
    ])
model.compile(optimizer='adam', loss='mae')

model.fit(
    train_dataset, epochs=settings["EPOCHS"],
    steps_per_epoch=settings["EVALUATION_INTERVAL"],
    validation_data=val_dataset, validation_steps=50
)

model.save("saved_model/model.h5")
tfjs.converters.save_keras_model(model, "saved_model/tfjs")
'''
for x, y in val_dataset:
    plot = show_plot([x[0].numpy(), y[0].numpy(), model.predict(x)[0]], 0, 'LSTM model')
    plot.show()
'''