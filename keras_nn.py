from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import sys
import math
from os.path import isfile, join, isdir
from os import listdir, makedirs
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization, InputLayer
from keras import regularizers
from keras.callbacks import EarlyStopping,ModelCheckpoint



N_epoch = 200
Plot = True
Load_binary = True
Beta = 0.01
N_patience = 8

Batch_size = 100
Time_steps = 120
Dimension = (80,)     # dimension for hidden layers       



global Capture_name
global parent_directory_binary
global parent_directory_text
global Write_binary

#parent_directory_text = 'leave-one-capture'
#parent_directory_binary = 'leave-one-capture' 
#Write_binary = False


def deep_nn(dimensions,n_features):
    n_layers = len(dimensions)

    model = Sequential()
    for i in range(n_layers):
        if i==0:
            model.add(InputLayer(input_shape=(n_features,)))
            model.add(BatchNormalization())
            
            model.add(Dense(dimensions[i], activation='tanh', 
                            kernel_initializer='glorot_uniform', #glorot
                            kernel_regularizer=regularizers.l2(Beta)))
            model.add(BatchNormalization())
            #model.add(Dropout(0.5))
        elif i==n_layers-1:
            model.add(Dense(dimensions[i],kernel_regularizer=regularizers.l2(Beta)))
        else:
            model.add(Dense(dimensions[i], activation='tanh', 
                            kernel_initializer='glorot_uniform', #glorot
                            kernel_regularizer=regularizers.l2(Beta)))
            model.add(BatchNormalization())
            #model.add(Dropout(0.5))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model



def save_plots(y,predict,folder_name):
    print('saving plots in ' + folder_name)
    for i in range(100):
        l1, = plt.plot(predict[i,:],'b')
        l2, = plt.plot(y[i,:],'r')
        plt.legend([l1, l2],['predicted', 'label'])
        plt.ylabel('Temperature')
        plt.xlabel('Time Steps')
        #plt.title('s={:.2f}, r={:.2f}, h={:.2f}'.format(scaled_X[i,0,0],scaled_X[i,0,1],scaled_X[i,0,-1]))
        directory = join(Capture_name, 'figures',folder_name)

        if not isdir(directory):
            makedirs(directory)

        plt.savefig(directory + '/{0}.png'.format(i))
        plt.close("all")



def roll_back(data, mean, std):
    if mean is None or std is None:
        return data

    new_data = data * std
    new_data += mean
    return new_data


def feature_scaling(data,mean=None,std=None):
    """
    data.shape must be (n_samples,n_features)
    return dataset with zero mean and unit variance
    """
    if mean is None:
        means = np.mean(data,axis=0)
    else:
        means = mean

    if std is None:
        stds = np.std(data,axis=0)
    else:
        stds = std

    new_data = data - means
    new_data = new_data / stds 
    return new_data,means,stds


def write_to_binary(data,fname):
    with open(fname, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    with open(fname, 'rb') as pickle_file:
        data2 = pickle.load(pickle_file)
    return np.array_equal(data,data2)



def main(_):

    start_time = time.time()

    global Capture_name

    
    binary_folder = 'binary_data/' + parent_directory_binary + '/' + Capture_name 

    if Write_binary:

        if not isdir(binary_folder):
            makedirs(binary_folder)

        input_dir_text = parent_directory_text + '/' + Capture_name    # change this if need
        print('going to read from ' + input_dir_text)

        # generate train inputs
        temp1 = np.loadtxt(input_dir_text + '/train_inputs.txt',delimiter=',')
        write_to_binary(temp1,binary_folder + '/train_inputs.p')

        # generate train labels
        temp1 = np.loadtxt(input_dir_text + '/train_labels.txt',delimiter=',')
        write_to_binary(temp1,binary_folder + '/train_labels.p')
        
        
        # generate test inputs
        temp1 = np.loadtxt(input_dir_text + '/test_inputs.txt',delimiter=',')
        write_to_binary(temp1,binary_folder + '/test_inputs.p')

        # generate test labels
        temp2 = np.loadtxt(input_dir_text + '/test_labels.txt',delimiter=',')
        write_to_binary(temp2,binary_folder + '/test_labels.p')
        
        return
    

    if Load_binary:
        # load train data
        print('load from {0}'.format(binary_folder))
        with open(binary_folder + '/train_inputs.p', 'rb') as pickle_file:
            X_train = pickle.load(pickle_file)
        with open(binary_folder + '/train_labels.p','rb') as pickle_file:
            y_train = pickle.load(pickle_file)

        # load test data
        with open(binary_folder + '/test_inputs.p', 'rb') as pickle_file:
            X_test = pickle.load(pickle_file)
        with open(binary_folder + '/test_labels.p','rb') as pickle_file:
            y_test = pickle.load(pickle_file)
        #return
    else:
        input_dir_text = parent_directory_text + '/' + Capture_name
        # load train data
        X_train = np.loadtxt(input_dir_text + '/train_inputs.txt',delimiter=',')
        y_train = np.loadtxt(input_dir_text + '/train_labels.txt',delimiter=',')

        # load test data
        X_test = np.loadtxt(input_dir_text + '/test_inputs.txt',delimiter=',')
        y_test = np.loadtxt(input_dir_text + '/test_labels.txt',delimiter=',')
        return



    Capture_name = join('result-'+parent_directory_text,Capture_name)

    if not isdir(Capture_name):
        makedirs(Capture_name)

    # save the test input and labels file
    if not isfile(Capture_name + '/test_inputs.txt'):
        np.savetxt(Capture_name + '/test_inputs.txt',X_test,fmt='%.6f',delimiter=',')
        np.savetxt(Capture_name + '/test_labels.txt',y_test,fmt='%.6f',delimiter=',')


    # scale features and labels
    mean=None
    std=None
    mean2=None
    std2=None
    #y_train,mean2,std2 = feature_scaling(y_train)
    #y_test,_,_ = feature_scaling(y_test,mean2,std2)

    
    N_Features = X_train.shape[-1]
    global Dimension
    Dimension += (y_train.shape[-1],)



    # split CV datasets
    X_train, X_cv, y_train, y_cv = train_test_split(X_train,y_train,test_size=0.2)


    # get starting temperature
    offset=Dimension[-1]-Time_steps
    if offset!=0:
        t_start_cv=X_cv[:,offset:];
        t_start_test=X_test[:,offset:]
        t_start_train=X_train[:,offset:]
        print('one sample of start T:')
        print(t_start_train[0,:])
    else:
        t_start_cv=None
        t_start_test=None
        t_start_train=None




    # load or write indices
    if not isfile(Capture_name + '/indices.txt'):
        max_idx2 = X_test.shape[0]
        idx2 = np.random.randint(0,high=max_idx2,size=(100,))
        np.savetxt(Capture_name + '/indices.txt',idx2,fmt='%i')
    else:
        idx2 = np.loadtxt(Capture_name + '/indices.txt',dtype=int)


    # create new folder
    Capture_name += '/'
    for i in range(len(Dimension)):
        if i != len(Dimension) - 1:
            Capture_name +=  '{0}_'.format(Dimension[i])
        else:
            Capture_name +=  '{0}'.format(Dimension[i])
    
    if not isdir(Capture_name):
        makedirs(Capture_name)

    print(Capture_name)

    # build model
    model=deep_nn(Dimension, N_Features)
    early_stop = EarlyStopping(monitor='val_loss', patience=N_patience)
    filepath = Capture_name + "/best-weights.hdf5"
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train, epochs=N_epoch, validation_data=(X_cv,y_cv),
                        batch_size=Batch_size, verbose=0, callbacks=[early_stop, checkpointer])
    #print(history.history.keys())

    # load weights
    model.load_weights(filepath)
    print("Created model and loaded weights from file")
    model.compile(loss='mean_squared_error', optimizer='adam')

    # make predictions
    predicts_train = model.predict(X_train)
    predicts_test = model.predict(X_test)
    predicts_cv = model.predict(X_cv)


    # roll back
    predicts_train = roll_back(predicts_train,mean2,std2)
    predicts_test = roll_back(predicts_test,mean2,std2)
    predicts_cv = roll_back(predicts_cv,mean2,std2)

    labels_train = roll_back(y_train,mean2,std2)
    labels_test = roll_back(y_test,mean2,std2)
    labels_cv = roll_back(y_cv,mean2,std2)



    # save all predicts to file
    np.savetxt(Capture_name + '/predicts_all.txt',predicts_test,fmt='%.6f',delimiter=',')


    mse_train=mean_squared_error(labels_train,predicts_train)
    mse_test=mean_squared_error(labels_test,predicts_test)
    mse_cv=mean_squared_error(labels_cv,predicts_cv)
    

    with open(Capture_name + '/loss.txt', 'w') as f:
        f.write('training loss: {0}\n'.format(mse_train))
        f.write('cv loss: {0}\n'.format(mse_cv))
        f.write('test loss: {0}\n'.format(mse_test))
    print('training loss: {0}'.format(mse_train))
    print('cv loss: {0}'.format(mse_cv))
    print('test loss: {0}\n'.format(mse_test))
    


    end_time = time.time()

    
    with open(Capture_name + '/loss.txt', 'a') as f:
        f.write('train time: {0}\n'.format(end_time - start_time))
    print('train time: {0}'.format(end_time - start_time))


    if not isdir(Capture_name + '/figures'):
        makedirs(Capture_name + '/figures')

    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(Capture_name+'/loss_history.png')
    plt.close("all")
    

    # cat starting T
    if t_start_cv is not None:
        predicts_cv = np.concatenate((t_start_cv,predicts_cv),axis=1)
        predicts_test = np.concatenate((t_start_test,predicts_test),axis=1)

        labels_cv = np.concatenate((t_start_cv,labels_cv),axis=1)
        labels_test = np.concatenate((t_start_test,labels_test),axis=1)


    if Plot:
        # randomly pick 100 samples
        max_idx = X_cv.shape[0]
        idx = np.random.randint(0,high=max_idx,size=(100,))

        labels_cv = labels_cv[idx,:]
        labels_test = labels_test[idx2,:]

        save_plots(labels_cv,predicts_cv[idx,:],'cv')
        save_plots(labels_test,predicts_test[idx2,:],'test')

    print('total time: {0}'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and test NN')

    parser.add_argument('parent_directory', type=str,
                       help='parent_drectory where has multiple capture folders')

    parser.add_argument('capture_name', type=str,
                       help='capture folder where has training and testing files')

    parser.add_argument('write_binary', type=int,
                       help='write training and testing to binary or not')


    args = parser.parse_args()



    parent_directory_binary=args.parent_directory
    parent_directory_text=args.parent_directory
    Capture_name=args.capture_name
    Write_binary=args.write_binary

    print(parent_directory_binary)
    print(Capture_name)
    print(Write_binary)

    tf.app.run(main=main)