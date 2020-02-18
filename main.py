import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


original = pd.read_csv("./data/lalavla-강남구-Nail.csv")
df = original[:576]

def univariate_data(dataset, start_index, end_index, history_size, target_size):
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

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
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

TRAIN_SPLIT = 400

#tf.random.set_seed(13)

uni_data = df["qty"]
uni_data.index = df["date"]

# Normalization
train_mean = uni_data[:TRAIN_SPLIT].mean()
train_std = uni_data[:TRAIN_SPLIT].std()

train_data = (uni_data - train_mean)/train_std
#data = tf.keras.utils.normalize(np.array(uni_data))

past_history = 32
future_target = 1

x_train, y_train = univariate_data(train_data, 0, TRAIN_SPLIT, past_history, future_target)
x_val, y_val = univariate_data(train_data, TRAIN_SPLIT, None, past_history, future_target)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_univariate = val_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
print(x_train.shape)
'''
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mae')

#for x, y in val_univariate.take(1):
#    print(model.predict(x).shape)

EVALUATION_INTERVAL = 20
EPOCHS = 1

model.fit(
    train_univariate, epochs=EPOCHS,
    steps_per_epoch=EVALUATION_INTERVAL,
    validation_data=val_univariate, validation_steps=50
)

#model.save('model.h5')
import tensorflowjs
tensorflowjs.converters.save_keras_model(model, "js")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_SELECT_TF_OPS]
converter.allow_custom_ops = True
tflite_model = converter.convert()

open("converted_model.tflite", "wb").wirte(tflite_model)


for x, y in val_univariate:
    plot = show_plot([x[0].numpy(), y[0].numpy(), model.predict(x)[0]], 0, 'LSTM model')
    plot.show()

test_df = original[500:500+past_history]

test_data = test_df["qty"]
test_data.index = test_df["date"]

print(test_data.head())

test_data = (test_data - train_mean)/train_std

test_data = np.reshape(test_data.to_numpy(), (1, past_history, 1))
print(test_data)
#test_univariate = tf.data.Dataset.from_tensor_slices((test_data))
print(model.predict(test_data))

#plot = show_plot([x_test[0].numpy(), y_test[0].numpy(), model.predict(x_test)[0]], 0, 'LSTM model')
#plot.show()
'''