import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from os import listdir,mkdir
from os.path import join,isdir 
from sklearn.model_selection import train_test_split
from PIL import Image
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model,Sequential # basic class for specifying and training a neural network
from keras.layers import *
from keras.utils import np_utils,to_categorical # utilities for one-hot encoding of ground truth values
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import metrics
from keras.optimizers import SGD
import utils


batch_size = 2 # in each iteration, we consider 32 training examples at once
num_epochs = 100 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
conv_depth_3 = 128
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons


n_fold = 11 # egg: 7, pancake: 6, bacon: 12|| for simpleLSTM pancake 11 bacon 8 egg 6
root_dir=r'D:/icme-breakfast/Miao_LSTM/Thermal_pancake/cnn_pancake_75'
output_dir=r'D:/icme-breakfast/Miao_LSTM/Miao_LSTM_cnn_pancake/results_bacon_75_e100_no_normalize_flir_regression_LSTMprediction_nolstmrms42'


# usually flir only
flir_only=True
rgb_only=False
is_merge=False
is_subtract=False


# usually not change
use_augment=False
is_save_imgs=True
is_scale=False
num_classes = 1


def deep_cnn(image_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=image_shape))
    model.add(BatchNormalization())

    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))

    '''
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    model.add(Flatten())

    model.add(Dense(hidden_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_prob_2))
    model.add(Dense(num_classes, activation='softmax'))
    '''

    model.add(Convolution2D(conv_depth_3, (kernel_size,kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_3, (1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(num_classes,(1, 1)))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    

    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam',
                  metrics=['accuracy',utils.top_2_accuracy]) 

    model.summary()
    return model



def deep_cnn2(image_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=image_shape))
    model.add(BatchNormalization())

    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # conv [128]
    model.add(Convolution2D(conv_depth_3, (kernel_size,kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_3, (1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Convolution2D(num_classes,(1, 1)))
    model.add(GlobalAveragePooling2D())
 

    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=[metrics.mean_absolute_error,metrics.mean_absolute_percentage_error, metrics.mean_squared_error])


    model.summary()
    return model



def conv_model(image_shape):
    num_iter=3

    inp=Input(shape=image_shape)
    x=BatchNormalization()(inp)

    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    for i in range(num_iter):
        x=Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same')(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)

    x=MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    x=Dropout(drop_prob_1)(x)


    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    for i in range(num_iter):
        x=Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same')(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)

    x=MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    x=Dropout(drop_prob_1)(x)


    # conv [128]
    for i in range(num_iter):
        if i != num_iter-1:
            x=Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same')(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)
        else:
            x=Convolution2D(conv_depth_3, (1,1))(x)
            x=BatchNormalization()(x)
            x=Activation('relu')(x)

    x=Convolution2D(num_classes,(1, 1))(x)
    x=GlobalAveragePooling2D()(x)

    return inp,x



def deep_cnn3(image_shape1,image_shape2):
    inp1,x1=conv_model(image_shape1)
    inp2,x2=conv_model(image_shape2)
    x=Concatenate()([x1, x2])
    out=Dense(num_classes, activation='softmax')(x)
    merged_model = Model([inp1, inp2], out)
    merged_model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam',
                  metrics=['accuracy',utils.top_2_accuracy]) 

    merged_model.summary()
    return merged_model
    


t1=time.time()

if not isdir(output_dir):
    mkdir(output_dir)



dicts_orig,dicts_aug=utils.load_imgs_from_folders(root_dir,use_augment,is_subtract,isregression=True)
dicts_orig,dicts_aug=utils.generate_n_fold_data(n_fold,dicts_orig,dicts_aug)
ks = dicts_orig.keys()
for k in ks:
    print('\ntarget key: {0}'.format(k))
    if use_augment:
        X_train,y_train,X_test,y_test=utils.generate_leave_one_data_all(dicts_orig,dicts_aug,k)
    else:
        X_train,y_train,X_test,y_test=utils.generate_leave_one_data_orig(dicts_orig,k)

    # split training and validation data
    X_train, X_cv, y_train, y_cv = train_test_split(X_train,y_train,test_size=0.2, random_state = 42)
    

    Y_train=y_train
    Y_test=y_test
    Y_cv=y_cv
    
    
    print('number of train samples: {0}'.format(X_train.shape[0]))
    print('number of cv samples: {0}'.format(X_cv.shape[0]))
    
    
    # seperate data
    X_train_rgb,X_train_flir=utils.rollback(X_train)
    X_cv_rgb,X_cv_flir=utils.rollback(X_cv)
    X_test_rgb,X_test_flir=utils.rollback(X_test)


    if is_scale:
        X_train_rgb=utils.scale_imarray(X_train_rgb)
        X_cv_rgb=utils.scale_imarray(X_cv_rgb)
        X_test_rgb=utils.scale_imarray(X_test_rgb)

        X_train_flir=utils.scale_imarray(X_train_flir,14)
        X_cv_flir=utils.scale_imarray(X_cv_flir,14)
        X_test_flir=utils.scale_imarray(X_test_flir,14)

        if (flir_only==False) and (rgb_only==False):
            X_train=np.append(X_train_rgb,X_train_flir,axis=-1)
            X_cv=np.append(X_cv_rgb,X_cv_flir,axis=-1)
            X_test=np.append(X_test_rgb,X_test_flir,axis=-1)


    if flir_only:
        X_train=X_train_flir
        X_cv=X_cv_flir
        X_test=X_test_flir
    if rgb_only:
        X_train=X_train_rgb
        X_cv=X_cv_rgb
        X_test=X_test_rgb

    
    
    if is_merge:
        X_train=[X_train_rgb,X_train_flir]
        X_cv=[X_cv_rgb,X_cv_flir]
        X_test=[X_test_rgb,X_test_flir]
        model=deep_cnn3(X_train_rgb.shape[1:],X_train_flir.shape[1:])
    else:
        model=deep_cnn2(X_train.shape[1:])
    
    
    

    if not isinstance(k,str):
        k=str(k)

    #"weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = join(output_dir,k+'-weights.hdf5')
    checkpointer = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=2, save_best_only=True)  
    
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                  verbose=2, validation_data=(X_cv,Y_cv), callbacks=[checkpointer])

    score1=model.evaluate(X_test, Y_test, verbose=2)
    print('\nresults of final epoch:')
    print('mae: {0}'.format(score1[1]))
    print('mape: {0}'.format(score1[2]))

    model.load_weights(filepath)
    print("loaded weights from file")

    score2=model.evaluate(X_test, Y_test, verbose=2)
    print('\nresults of best val:')
    print('mae: {0}'.format(score2[1]))
    print('mape: {0}'.format(score2[2]))

    y_predict=model.predict(X_test)
    np.savetxt( join(output_dir,'{0}-labels.txt'.format(k)),y_test,fmt='%.4f',delimiter=',' )
    np.savetxt( join(output_dir,'{0}-predicts.txt'.format(k)),y_predict,fmt='%.4f',delimiter=',' )


    with open(join(output_dir,'score.txt'),'a') as f:
        mse1 = round(score1[3],4)
        mse2 = round(score2[3],4)
        mape1 = round(score1[2],2)
        mape2 = round(score2[2],2)
        mae1 = round(score1[1],4)
        mae2 = round(score2[1],4)
        s = "{0}: mape: {1} {2} ; mae: {3} {4}; mse {5} {6}\n".format(k,mape1,mape2,mae1,mae2, mse1, mse2)
        f.write(s)

    
    if is_save_imgs:
        # scale test data
        if not is_scale:
            X_test_rgb=utils.scale_imarray(X_test_rgb)
            X_test_flir=utils.scale_imarray(X_test_flir,14)

            if (flir_only==False) and (rgb_only==False):
                X_test2=np.append(X_test_rgb,X_test_flir,axis=-1)
            if flir_only:
                X_test2=X_test_flir
            if rgb_only or is_merge:
                X_test2=X_test_rgb

        # save imgs
        print('saving test images...')
        output_imdir=join(output_dir,k+'-imgs')
        if not isdir(output_imdir):
            mkdir(output_imdir)
        utils.save_imarray(X_test2,output_imdir,Y_test,y_predict,precision=4)

    
    del X_train,X_cv,X_train_rgb,X_train_flir,X_cv_flir,X_cv_rgb
    t2=time.time()
    print('total time: {0}'.format(t2-t1))

