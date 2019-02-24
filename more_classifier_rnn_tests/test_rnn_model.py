#C:\Users\Raghav Ganesh\AppData\Local\Programs\Python\Python35\Scripts\pip.exe

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(3)

import numpy as np
from numpy import newaxis
from pandas import read_csv
#from sklearn.preprocessing import MinMaxScaler

from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed
import sys
import math
import os
import time
import warnings
import pprint

EPOCHS = 10

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
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #eda_val = scaler.fit_transform(eda_val)

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
Divide a dataset into chunks for statistics

'''

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_model(steps_before, steps_after, feature_count):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    DROPOUT = 0.5
    LAYERS = 2
    
    hidden_neurons = 300

    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    model.add(RepeatVector(steps_after))
    model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))  
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    return model


def train_pred(model, dataX, dataY, epoch_count):
    """ 
        trains only the pred model
    """
    history = model.fit(dataX, dataY, batch_size=1, nb_epoch=epoch_count, validation_split=0.05)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    input("dab")

def run():
    ''' 
        Used Elements of pred Model Example
    '''
    print('Preprocessing Data')

    rootdir = './E4_data/'
    files_val = []
    # Iterate through all files/subdirectories in 'E4_data'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if (os.path.join(subdir, file).find('EDA') > -1) and (os.path.join(subdir, file).find('pdf')) == -1:
                files_val.append(os.path.join(subdir, file))


    print(str(len(files_val)) + " Files to parse");
    # Resampling with 27 datasets for train and 2 datasets for test
    g_iter = 0
    
    # Categorize Data in Dataset
    eda_vals = []
    aggression_vals = []

    #for iter in range(0,28):
    #for j in range(iter+1,28):
    train_files = []
    for file in files_val:
        train_files.append(file)

    #test_flat = [train_files[iter],train_files[j]]
    #del train_files[iter]
    #del train_files[j]

    for one_file in train_files:
        # Parse dataset
        raw_eda_val, eda_val, labels, behavior_val, aggression_eda = load_dataset(one_file)
        for val in eda_val:
            eda_vals.append(val)
        for val in aggression_eda:
            aggression_vals.append(val)


    t = np.arange(0.0, 4.0, 0.02)
    #pred = np.sin(2 * np.pi * t)
    #pred = pred.reshape((pred.shape[0], 1))
    
    n_pre = 75
    n_post = 25

    pred = np.asarray(aggression_vals)
    pred = pred[:-200]
    pred = pred.reshape((pred.shape[0], 1))

    dX, dY = [], []
    for i in range(len(pred)-n_pre-n_post):
        dX.append(pred[i:i+n_pre])
        dY.append(pred[i+n_pre:i+n_pre+n_post])
        #dY.append(pred[i+n_pre])
    dataX = np.array(dX)
    dataY = np.array(dY)

    # create and fit the LSTM network
    print('creating model...')
    model = create_model(n_pre, n_post, 1)
    train_pred(model, dataX, dataY, EPOCHS)
    
    # now test
    t = np.arange(15.0, 19.0, 0.02)
    #pred = np.sin(2 * np.pi * t)
    #pred = pred.reshape((pred.shape[0], 1))
    pred = np.loadtxt('calm.csv',delimiter=",",skiprows=1,usecols=0)
    pred = np.delete(pred,np.s_[0:299])
    pred = pred.reshape((pred.shape[0], 1))
    
    dX, dY = [], []
    for i in range(len(pred)-n_pre-n_post):
        dX.append(pred[i:i+n_pre])
        dY.append(pred[i+n_pre:i+n_pre+n_post])
    dataX = np.array(dX)
    dataY = np.array(dY)
    
    predict = model.predict(dataX)
    
    # now plot
    nan_array = np.empty((n_pre - 1))
    nan_array.fill(np.nan)
    nan_array2 = np.empty(n_post)
    nan_array2.fill(np.nan)
    ind = np.arange(n_pre + n_post)

    fig, ax = plt.subplots()
    for i in range(0, 50, 50):

        forecasts = np.concatenate((nan_array, dataX[i, -1:, 0], predict[i, :, 0]))
        ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0], dataY[i, :, 0]))
        network_input = np.concatenate((dataX[i, :, 0], nan_array2))
     
        ax.plot(ind, network_input, 'b-x', label='Network input')
        ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
        ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
        
        plt.xlabel('Time')
        plt.ylabel('EDA')
        plt.title('Many to Many Forecast')
        plt.legend(loc='best')
        plt.savefig('calm_with_stress/plot_mtm_triple_' + str(i) + '.png')
        plt.cla()

def main():
    run()
    return 1

if __name__ == "__main__":
    sys.exit(main())
