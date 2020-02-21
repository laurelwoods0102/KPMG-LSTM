import numpy as np
import pandas as pd
import json
import datetime

import tensorflow as tf

class Preprocess():
    def __init__(self, data):
        with open("settings.json") as j:
            settings = json.load(j)
        self.data = data
        self.test_split = settings["TEST_SPLIT"]
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
        
    def create_train_dataset(self, BUFFER_SIZE=50000):
        # if training, change None -> self.train_split
        x_train, y_train = self.split_data(self.dataset, 0, None, self.past_history, self.future_target)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(self.batch_size).repeat()

        return train_dataset, x_train.shape
    
    def create_val_dataset(self, BUFFER_SIZE=50000):
        x_val, y_val = self.split_data(self.dataset, self.train_split, self.test_split, self.past_history, self.future_target)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.cache().shuffle(BUFFER_SIZE).batch(self.batch_size).repeat()

        return val_dataset

    
    def create_test_dataset(self, BUFFER_SIZE=50000):
        x_test, y_test = self.split_data(self.dataset, self.test_split, None, self.past_history, self.future_target)
        self.create_predict_dataset(x_test[-(self.future_target):])

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.cache().batch(self.batch_size)

        return test_dataset
    
    def predict_date_min(self):
        latest_date = datetime.datetime.strptime(str(self.dataset.index[-1]), r'%Y%m%d')
        min_available = latest_date + datetime.timedelta(days=1)

        return min_available

    def create_predict_dataset(self, x):
        date = self.predict_date_min()

        for i in range(7):
            x_elem = x[i].reshape(1, 32)
            x_date = (date + datetime.timedelta(days=i)).strftime(r'%Y%m%d')

            with open("predict_dataset/{}.json".format(x_date), "w") as j:
                json.dump(x_elem.tolist()[0], j, indent=4)
            np.savetxt('predict_dataset/csv/{}.csv'.format(x_date), x_elem.T, delimiter=", ")

    
if __name__ == "__main__":    
    dataframe = pd.read_csv("./data/dataset.csv")
    data = dataframe["qty"]
    data.index = dataframe["date"]

    preprocess = Preprocess(data)
    mean, std = preprocess.normalization()
    '''
    train_dataset, val_dataset, shape = preprocess.create_dataset()

    print(shape)
    for x, y in train_dataset.take(1):
        print(x)
        print("------------------------------------------")
        print(y)
    '''
    normalize = np.array([mean, std])
    np.savetxt('normalize.csv', normalize, delimiter=", ")
    
    #preprocess.create_test_dataset()
    #preprocess.create_test_dataset()