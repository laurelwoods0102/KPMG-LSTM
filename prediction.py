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

df = pd.read_csv("data/prediction_20190101.csv")
data = df["qty"].to_numpy()
data = np.reshape(data, (1, 32, 1))

normalize = pd.read_csv("normalize.csv", header=None).to_numpy().reshape(1, 2)[0]

model = keras.models.load_model("saved_model/model.h5")
print(model.predict(data)[0][0] * normalize[1] + normalize[0])
