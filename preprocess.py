import numpy as np
import pandas as pd
import json

import tensorflow as tf

class Preprocess():
    def __init__(self, data):
        with open("settings.json") as j:
            settings = json.load(j)
        self.data = data
        self.train_split = settings["TRAIN_SPLIT"]
        self.past_history = settings["PAST_HISTORY"]
        self.future_target = settings["FUTURE_TARGET"]
        self.batch_size = settings["BATCH_SIZE"]
        
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
        
    def create_dataset(self, BUFFER_SIZE=50000):
        x_train, y_train = self.split_data(self.dataset, 0, self.train_split, self.past_history, self.future_target)
        x_val, y_val = self.split_data(self.dataset, self.train_split, None, self.past_history, self.future_target)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(self.batch_size).repeat()

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.cache().shuffle(BUFFER_SIZE).batch(self.batch_size).repeat()

        return train_dataset, val_dataset, x_train.shape
    
if __name__ == "__main__":
    dataframe = pd.read_csv("./data/lalavla-강남구-Nail.csv")
    dataframe = dataframe[:576]

    data = dataframe["qty"]
    data.index = dataframe["date"]

    preprocess = Preprocess(data)
    mean, std = preprocess.normalization()
    train_dataset, val_dataset, shape = preprocess.create_dataset()

    print(shape)
    print(train_dataset)
