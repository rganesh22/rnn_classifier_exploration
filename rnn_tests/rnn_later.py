import math
import os
import time
import warnings

import numpy as np
#import tensorflow as tf
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import newaxis
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

#tf.set_random_seed(14)
np.random.seed(14)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

'''
Function that reads one CSV file and returns the contents in multiple lists
And numpy arrays

'''

def load_dataset(file_path):

    csv_data = np.genfromtxt(file_path, dtype=str, delimiter=",")

    raw_eda_val = np.asarray(read_csv(file_path, usecols=[0], engine='python'))
    eda_val = raw_eda_val
    
    timestamps_val = csv_data[:,1]
    behavior_val = csv_data[:,2]

    aggression_eda_val = []
    aggression_timestamps_val = []

    for i in range(len(behavior_val)-2,-1,-1):
        if behavior_val[i] != 'NA' or behavior_val[i] != 'sleeping':
            aggression_eda_val.append(eda_val[i])
            np.delete(eda_val,i,axis=0)

            aggression_timestamps_val.append(timestamps_val[i])
    
    labels = []
    for i in range(0,len(eda_val) - 1):
        labels.append(i)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    eda_val = scaler.fit_transform(eda_val)

    return raw_eda_val, eda_val.astype(float), labels, behavior_val, aggression_eda_val

'''
Convert a dataset to the right format
Used in 'test_train()' function

'''

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

'''
Divide a dataset into testing and training segments

'''

def test_train(dataset):

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    return trainX, trainY, testX, testY

'''
Create LSTM RNN Neural Network Model

'''

def create_model():
    hidden = 128
    model = Sequential()

    model.add(LSTM(input_dim=1, output_dim=hidden))
    model.add(Dropout(.2))
    model.add(Dense(input_dim=hidden, output_dim=1))

    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    
    return model

'''
Sliding Window Predictor

''' 

def predict_sliding_window(model,data,pred_len):
    
    mod_data = data
    predictions = []
    for i in range(0,pred_len):
        predict = model.predict(mod_data)
        predictions.append(predict[len(predict)-1])

        #print(mod_data.shape)
        mod_data = np.delete(mod_data,[0,0,0],axis=0)

        pred = np.array(predict[len(predict)-1])

        pred_array = np.empty([1, 1])
        pred_array[0,0] = pred

        #print(pred_array)

        mod_data = np.append(mod_data,pred_array)
        mod_data = np.reshape(mod_data,(len(mod_data),1,1))
        #print(mod_data.shape)

    return predictions
