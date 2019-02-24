import math
import os
import time
import warnings

import numpy as np
#from keras.layers.core import Activation, Dense, Dropout
#from keras.layers.recurrent import LSTM
#from keras.models import Sequential
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

if __name__=='__main__':
        global_start_time = time.time()
        print('Preprocessing Data')

        right = 0
        wrong = 0

        g_error = []
        g_a_error = []

        rootdir = './E4_data/'
        files_val = []
        # Iterate through all files/subdirectories in 'E4_data'
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if (os.path.join(subdir, file).find('EDA') > -1) and (os.path.join(subdir, file).find('pdf')) == -1:
                    files_val.append(os.path.join(subdir, file))

        # Resampling with 27 datasets for train and 2 datasets for test
        g_iter = 0
        for iter in range(0,28):
            for j in range(iter+1,28):
                train_files = []
                for file in files_val:
                    train_files.append(file)

                #print(len(files_val))
                #print(len(train_files))
                test_flat = [train_files[iter],train_files[j]]
                del train_files[iter]
                del train_files[j]
                #print('i: ', iter)
                #print('j: ', j)

                # Categorize Data in Dataset
                eda_vals = []
                aggression_vals = []
                for one_file in train_files:
                    # Parse dataset
                    raw_eda_val, eda_val, labels, behavior_val, aggression_eda = load_dataset(one_file)
                    for val in eda_val:
                        eda_vals.append(val)
                    for val in aggression_eda:
                        aggression_vals.append(val)
                    
                behavior_vals = []
                raw_eda_vals = []
                for test_file in test_flat:
                    raw_eda_val, eda_val, labels, behavior_val, aggression_eda = load_dataset(one_file)
                    
                    for val in behavior_val:
                        behavior_vals.append(val)

                    for val in raw_eda_val:
                        raw_eda_vals.append(val)

                trainX, trainY, testX, testY = test_train(eda_vals)
                a_trainX, a_trainY, a_testX, a_testY = test_train(aggression_vals)
                val_trainX, val_trainY, val_testX, val_testY = test_train(raw_eda_vals)

                # Reshape Data
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

                a_trainX = np.reshape(a_trainX, (a_trainX.shape[0], 1, a_trainX.shape[1]))
                a_testX = np.reshape(a_testX, (a_testX.shape[0], 1, a_testX.shape[1]))

                val_trainX = np.reshape(val_trainX, (val_trainX.shape[0], 1, val_trainX.shape[1]))
                val_testX = np.reshape(val_testX, (val_testX.shape[0], 1, val_testX.shape[1]))
                print('Shape of trainX is ' + str(trainX.shape))
                print('Shape of a_trainX is ' + str(a_trainX.shape))
                print('Done Preprocessing Data')

