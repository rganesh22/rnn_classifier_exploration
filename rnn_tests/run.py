import rnn
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import math
import operator


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
                    raw_eda_val, eda_val, labels, behavior_val, aggression_eda = rnn.load_dataset(one_file)
                    for val in eda_val:
                        eda_vals.append(val)
                    for val in aggression_eda:
                        aggression_vals.append(val)
                    
                behavior_vals = []
                raw_eda_vals = []
                for test_file in test_flat:
                    raw_eda_val, eda_val, labels, behavior_val, aggression_eda = rnn.load_dataset(one_file)
                    
                    for val in behavior_val:
                        behavior_vals.append(val)

                    for val in raw_eda_val:
                        raw_eda_vals.append(val)

                trainX, trainY, testX, testY = rnn.test_train(eda_vals)
                a_trainX, a_trainY, a_testX, a_testY = rnn.test_train(aggression_vals)
                val_trainX, val_trainY, val_testX, val_testY = rnn.test_train(raw_eda_vals)

                # Reshape Data
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

                a_trainX = np.reshape(a_trainX, (a_trainX.shape[0], 1, a_trainX.shape[1]))
                a_testX = np.reshape(a_testX, (a_testX.shape[0], 1, a_testX.shape[1]))

                val_trainX = np.reshape(val_trainX, (val_trainX.shape[0], 1, val_trainX.shape[1]))
                val_testX = np.reshape(val_testX, (val_testX.shape[0], 1, val_testX.shape[1]))
                print('Done Preprocessing Data')

                print('Creating RNN Model')
                start = time.time()
                rnn_model = rnn.create_model()
                a_rnn_model = rnn.create_model()

                # Fit Model
                epochs = 10
                rnn_model.fit(
                    trainX,
                    trainY,
                    batch_size=512,
                    nb_epoch=epochs,
                    validation_split=0.05)
                
                epochs = 10
                a_rnn_model.fit(
                    a_trainX,
                    a_trainY,
                    batch_size=512,
                    nb_epoch=epochs,
                    validation_split=0.05)

                print('Done Creating RNN Model')
                print('Model Compilation Time : ', time.time() - start)
                
                #predict = rnn_model.predict(val_trainX[start:end])
                #print(predict)
                
                
                pred_time = time.time()
                start = 0
                end = 240
                pred_len = 240
                predictions = rnn.predict_sliding_window(rnn_model,val_trainX[start:end],pred_len)
                a_predictions = rnn.predict_sliding_window(a_rnn_model,val_trainX[start:end],pred_len)
                print('Prediction Time',time.time() - start)
                

                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(eda_val)

                s_a_predictions = scaler.inverse_transform(a_predictions)
                s_predictions = scaler.inverse_transform(predictions)

                #print(len(s_a_predictions))
                #print(len(val_trainY))

                a_error = 0
                error = 0
                sum = 0
                for i in range(start,end):
                    sum = sum + (val_trainY[i+pred_len] - s_a_predictions[i])
                    
                a_error = math.sqrt(math.pow(sum,2))

                sum = 0
                for i in range(start,end):
                    sum = sum + (val_trainY[i+pred_len] - s_predictions[i])
                
                error = math.sqrt(math.pow(sum,2))

                print('Aggression Classifier Error: ' + str(a_error))
                print('Non-Aggression Classifier Error: ' + str(error))

                g_error.append(error)
                g_a_error.append(a_error)

                correct = 1
                occur = 0
                for val in behavior_vals[start+2*pred_len:end+2*pred_len]:
                    if val != 'NA' or val != 'sleeping':
                        #occur = occur + 1
                        correct = 0
                
                #if float(occur/(end+pred_len - start+pred_len)) >= float(1/2):
                #    correct = 0

                index, value = min(enumerate([a_error,error]), key=operator.itemgetter(1))

                if index == correct:
                    right = right + 1
                    print('Right')
                else:
                    wrong = wrong + 1
                    print('Wrong')

                
                #plt.plot(np.reshape(testX,(testX.shape[0]))[20:150])
                #'''
                plt.plot(testY)
                plt.plot(s_a_predictions)
                plt.plot(s_predictions)
                plt.show()
                #''
                print('Training duration (s) : ', time.time() - global_start_time)
                print('Iterations:', g_iter)
                g_iter = g_iter + 1
                print('Overall Right (In-Loop):' + str(right))
                print('Overall Wrong (In-Loop):' + str(wrong))

print('Overall Right (Final):' + str(right))
print('Overall Wrong (Final):' + str(wrong))

print('Overall Error Aggression:' + str(g_a_error))
print('Overall Error Non-Aggression:' + str(g_error))
