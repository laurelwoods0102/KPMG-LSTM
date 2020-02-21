import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import shutil

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
dataframe = pd.read_csv("./data/dataset.csv")

data = dataframe["qty"]
data.index = dataframe["date"]

preprocess = Preprocess(data)
mean, std = preprocess.normalization()
train_dataset, shape = preprocess.create_train_dataset()
#val_dataset = preprocess.create_val_dataset()

normalize = np.array([mean, std])
if os.path.isfile("normalize.csv"):
    os.remove("normalize.csv")
np.savetxt('normalize.csv', normalize, delimiter=", ")

# Model
'''
model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, input_shape=shape[1:]),     # Pylint error
        tf.keras.layers.Dense(1)
    ])
'''
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='softsign', input_shape=shape[1:])),
    tf.keras.layers.Dense(settings["FUTURE_TARGET"])
])

model.compile(optimizer='adam', loss='mae')
'''
model.fit(
    train_dataset, epochs=settings["EPOCHS"],
    steps_per_epoch=settings["EVALUATION_INTERVAL"],
    validation_data=val_dataset, validation_steps=50
)
'''
model.fit(
    train_dataset, epochs=settings["EPOCHS"],
    steps_per_epoch=settings["EVALUATION_INTERVAL"]
)
'''
test_dataset = preprocess.create_test_dataset()

model = keras.models.load_model("saved_model/model.h5")

for x, y in test_dataset:
    results = model.evaluate(x, y, batch_size=settings["BATCH_SIZE"])
    print('test loss, test acc:', results)
'''
model.save("saved_model/model.h5")
tfjs.converters.save_keras_model(model, "saved_model/tfjs")
