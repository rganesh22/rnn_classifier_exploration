import math
import os
import time
import warnings
import pprint

import numpy as np
from numpy import newaxis
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
Divide a dataset into chunks for statistics

'''

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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

        rootdir = './E4_data/'
        files_val = []
        # Iterate through all files/subdirectories in 'E4_data'
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if (os.path.join(subdir, file).find('ACC') > -1) and (os.path.join(subdir, file).find('pdf')) == -1:
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
            
        behavior_vals = []
        raw_eda_vals = []
        for test_file in train_files:
            raw_eda_val, eda_val, labels, behavior_val, aggression_eda = load_dataset(one_file)
            
            for val in behavior_val:
                behavior_vals.append(val)

            for val in raw_eda_val:
                raw_eda_vals.append(val)



        chunked_eda_Vals = list(chunks(eda_vals, 20))
        chunked_aggression_Vals = list(chunks(aggression_vals, 20))


        labels = []
        statistic_eda_list = []
        for cluster_eda in chunked_eda_Vals:
            mean = np.mean(np.asarray(cluster_eda))
            std = np.std(np.asarray(cluster_eda));
            statistic_eda_list.append([mean,std])
            labels.append(0)

        statistic_aggression_list = []
        for cluster_eda in chunked_aggression_Vals:
            mean = np.mean(np.asarray(cluster_eda))
            std = np.std(np.asarray(cluster_eda));
            statistic_aggression_list.append([mean,std]);
            labels.append(1)
            #input("actually here");
        statistic_concatenated_list = statistic_eda_list + statistic_aggression_list

        s_trainX, s_testX, s_trainY, s_testY = train_test_split(statistic_concatenated_list, labels, test_size=0.33, random_state=42)

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        for name,clf in zip(names,classifiers):
            clf.fit(s_trainX,s_trainY)
            score = clf.score(s_testX,s_testY)
            print('Accuracy of ' + name + ': ' + str(score))

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
        

        '''
        Save Collated Files
        '''
        np.savetxt("processed\\trainx.csv",np.squeeze(trainX), delimiter=",");
        np.savetxt("processed\\ttrainy.csv",np.squeeze(trainY), delimiter=",");

        np.savetxt("processed\\testx.csv",np.squeeze(testX), delimiter=",");
        np.savetxt("processed\\testy.csv",np.squeeze(testY), delimiter=",");

        np.savetxt("processed\\a_trainx.csv",np.squeeze(a_trainX), delimiter=",");
        np.savetxt("processed\\a_trainy.csv",np.squeeze(a_trainY), delimiter=",");

        np.savetxt("processed\\a_testx.csv",np.squeeze(a_testX), delimiter=",");
        np.savetxt("processed\\a_testy.csv",np.squeeze(a_testY), delimiter=",");
        
        np.savetxt("processed\\val_trainx.csv",np.squeeze(val_trainX), delimiter=",");
        np.savetxt("processed\\val_trainy.csv",np.squeeze(val_trainY), delimiter=",");

        np.savetxt("processed\\val_testx.csv",np.squeeze(val_testX), delimiter=",");
        np.savetxt("processed\\val_testy.csv",np.squeeze(val_testY), delimiter=",");

        print('Shape of trainX is ' + str(trainX.shape))
        print('Shape of a_trainX is ' + str(a_trainX.shape))
        print('Execution time ' + str(time.time() - global_start_time))
        print('Done Preprocessing Data')

