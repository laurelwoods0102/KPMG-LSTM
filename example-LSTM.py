import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_csv("./data/lalavla-강남구-Nail.csv")
df = df[:500]

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = list()
    labels = list()

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)

TRAIN_SPLIT = 400

tf.random.set_seed(13)

uni_data = df["qty"]
uni_data.index = df["date"]

# Normalization
train_mean = uni_data[:TRAIN_SPLIT].mean()
train_std = uni_data[:TRAIN_SPLIT].std()
print(train_mean, train_std)

train_data = (uni_data - train_mean)/train_std
#data = tf.keras.utils.normalize(np.array(uni_data))
print(train_data)

past_history = 32
future_target = 0

x_train, y_train = univariate_data(train_data, 0, TRAIN_SPLIT, past_history, future_target)
x_val, y_val = univariate_data(train_data, TRAIN_SPLIT, None, past_history, future_target)