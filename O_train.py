import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras

class Preprocess():
    def __init__(self, data, train_split, past_history, future_target, batch_size, evaluation_interval, epochs, seed=None):        
        if seed:
            tf.random.set_seed(seed)
        self.data = data
        self.train_split = train_split
        self.past_history = past_history
        self.future_target = future_target
        self.batch_size = batch_size
        self.evaluation_interval = evaluation_interval
        self.epochs = epochs
        
    def normalization(self):    # Normalization
        train_mean = self.data.mean()
        train_std = self.data.std()

        self.dataset = (self.data - train_mean)/train_std
        
        return train_mean, train_std
    
    def split_data(self, dataset, start_index, end_index, history_size, target_size):
        data = list()
        labels = list()

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        
        for i in range(start_index, end_index):
            indices = dataset[i-history_size:i].to_numpy()
            data.append(np.reshape(indices, (history_size, 1)))
            y = dataset[i:i + target_size]
            labels.append(y.values)
        return np.array(data), np.array(labels)

    def create_time_steps(self, length):
        return list(range(-length, 0))

    def show_plot(self, plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = self.create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                        label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future+5)*2])
        plt.xlabel('Time-Step')

        return plt

    def create_dataset(self, BUFFER_SIZE=50000):
        x_train, y_train = self.split_data(self.dataset, 0, TRAIN_SPLIT, self.past_history, self.future_target)
        x_val, y_val = self.split_data(self.dataset, TRAIN_SPLIT, None, self.past_history, self.future_target)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_dataset = val_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        self.shape = x_train.shape
    
def create_model(self):
    self.model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, input_shape=self.shape[1:]),
        tf.keras.layers.Dense(1)
    ])
    self.model.compile(optimizer='adam', loss='mae')

if __name__ == "__main__":
    # Hyper-parameters
    TRAIN_SPLIT = 500
    PAST_HISTORY = 32
    FUTURE_TARGET = 1
    BATCH_SIZE = 64
    EVALUATION_INTERVAL = 200
    EPOCHS = 10

    dataframe = pd.read_csv("./data/lalavla-강남구-Nail.csv")
    dataframe = dataframe[:576]

    data = dataframe["qty"]
    data.index = dataframe["date"]
    
    preprocess = Preprocess(data, TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET, BATCH_SIZE, EVALUATION_INTERVAL, EPOCHS)
    
    mean, std = preprocess.normalization()
    preprocess.create_dataset()
