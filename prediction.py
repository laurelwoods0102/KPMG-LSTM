import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

from preprocess import Preprocess
from plotting import show_plot


class Prediction:
    def __init__(self):
        #with open("settings.json") as j:
        #    self.past_history = json.load(j)["PAST_HISTORY"]
        #self.dataset = df[-(self.past_history + 31):]

        self.normals = pd.read_csv("normalize.csv", header=None).to_numpy().reshape(1, 2)[0]
        self.model = keras.models.load_model("saved_model/model.h5")
    '''
    def available_predict(self):
        latest_date = datetime.datetime.strptime(str(self.dataset.iloc[-1]["date"]), r'%Y%m%d')
        min_available = latest_date + datetime.timedelta(days=1)
        max_available = latest_date + datetime.timedelta(days=self.past_history)

        return min_available.strftime(r'%Y%m%d'), max_available.strftime(r'%Y%m%d')
    '''        

    def predict(self, data):
        return int(self.model.predict(data)[0][0] * self.normals[1] + self.normals[0])

if __name__ == "__main__":
    df = pd.read_csv("data/prediction_20190101.csv")
    data = df["qty"].to_numpy()
    data = np.reshape(data, (1, 32, 1))

    predictor = Prediction()
    print(predictor.predict(data))
