import numpy as np
import numpy.matlib
import scipy.io as sio
from natsort import natsorted
from PIL import Image
from random import shuffle,seed
from os import listdir,mkdir
from os.path import join,isdir,isfile 
import matplotlib.pyplot as plt
from keras.metrics import top_k_categorical_accuracy
import copy
from sklearn.model_selection import train_test_split
import scipy.io as sio


def rgb2gray(np_imgs):
    """
    convert a 3D image or a 4D numpy array (channel last) to grayscale
    return a grayscale 4D numpy array
    """
    gray_imgs = np.dot(np_imgs,[0.299, 0.587, 0.114])
    if len(gray_imgs.shape) < 4:
        gray_imgs = gray_imgs[...,np.newaxis]
    return gray_imgs



def load_imgs(folder):

    imgs_list = []
    lists = listdir(folder)
    lists = natsorted(lists)
    
    for f in lists:
        fname = join(folder,f)

        if isdir(fname):
            continue

        img = Image.open(fname)
        imgarray = np.array(img,dtype=np.uint16)
        imgs_list.append(imgarray)

    np_imgs = np.array(imgs_list)
    if len(np_imgs.shape) < 4:
        np_imgs = np_imgs[...,np.newaxis]

    return np_imgs



def load_soap_imgs(folder):
    """
    load soap imgs from the folder and return dictionary 
    dict's keys: sub folder name
    dict's values: [imgs labels]
    """
    lists = listdir(folder)
    dict = {}
    for f in lists:

        l_split = f.split('_')
        f_type = l_split[0]
        print('key: {0}'.format(f_type))

        if f_type=='above':
            label=1
        elif f_type=='below':
            label=0
        
        subf = join(folder,f)
        imgs = load_imgs(subf)

        labels = np.ones((imgs.shape[0],)) * label
        keys = dict.keys()

        if f_type not in keys:
            dict[f_type] = [imgs,labels]
        else:
            dict[f_type][0] = np.append(dict[f_type][0],imgs,axis=0)
            dict[f_type][1] = np.append(dict[f_type][1],labels,axis=0)
    return dict



def load_beans_imgs_split_by_precent(folder,percent,is_orig=False):
    """
    load beans imgs from the folder
    and split the train and test by percentage
    """

    
    test_imgs = None
    test_labels = None

    lists = listdir(folder)
    dict = {} # for training
    for i,f in enumerate(lists):
        l_split = f.split('-')
        bean_type = l_split[0]
        label_pattern = l_split[-1]

        l_str = label_pattern.split('l')[-1]
        label = int(l_str)

        subf = join(folder,f)
        imgs = load_imgs(subf)
        labels = np.ones((imgs.shape[0],)) * label

        #split data by percentage
        num_train = imgs.shape[0] * percent
        assert(np.floor(num_train) == np.array(num_train))
        num_train = int(num_train)

        if i == 0:
            print('num of train: {0}'.format(num_train))

        if test_imgs is None:
            test_imgs = imgs[num_train:]
            test_labels = labels[num_train:]
        else:
            test_imgs = np.append(test_imgs,imgs[num_train:],axis=0)
            test_labels = np.append(test_labels,labels[num_train:],axis=0)
        
        keys = dict.keys()

        if bean_type not in keys:
            dict[bean_type] = [imgs[0:num_train],labels[0:num_train]]
        else:
            dict[bean_type][0] = np.append(dict[bean_type][0],imgs[0:num_train],axis=0)
            dict[bean_type][1] = np.append(dict[bean_type][1],labels[0:num_train],axis=0)
    
    if is_orig:
        test_data = [test_imgs,test_labels]
    else:
        test_data = None
    return dict,test_data
        

        


def load_beans_imgs(folder):
    """
    load beans imgs from the folder and return dictionary 
    dict's keys: bean types
    dict's values: [imgs labels]
    """
    lists = listdir(folder)
    dict = {}
    for f in lists:

        l_split = f.split('-')
        bean_type = l_split[0]
        label_pattern = l_split[-1]

        l_str = label_pattern.split('l')[-1]
        label = int(l_str)

        
        subf = join(folder,f)
        imgs = load_imgs(subf)

        labels = np.ones((imgs.shape[0],)) * label
        keys = dict.keys()

        if bean_type not in keys:
            dict[bean_type] = [imgs,labels]
        else:
            dict[bean_type][0] = np.append(dict[bean_type][0],imgs,axis=0)
            dict[bean_type][1] = np.append(dict[bean_type][1],labels,axis=0)
    return dict



def generate_beans_transfer_data(dict_orig,dict_aug, train_tuple, test_tuple):

    ks = dict_aug.keys()
    y_train = None
    X_train = None
    X_test = None
    for k in ks:
        data_aug = dict_aug[k]
        data_orig = dict_orig[k]

        # scale data
        rgb_imgs_aug = data_aug[0]
        rgb_imgs_orig = data_orig[0]
        
        
        labels_aug = data_aug[1]
        labels_orig = data_orig[1]

        if k in train_tuple:
            print('train: {0}'.format(k))

            # cat as 4th channel
            if X_train is None:
                X_train = rgb_imgs_aug
                y_train = labels_aug
            else:
                X_train = np.append(X_train, rgb_imgs_aug,axis = 0)
                y_train = np.append(y_train,labels_aug, axis = 0)    

            print('test: {0}'.format(k))
            if X_test is None:
                X_test = rgb_imgs_orig
                y_test = labels_orig
                
            else:
                X_test = np.append(X_test,rgb_imgs_orig,axis=0)
                y_test = np.append(y_test,labels_orig,axis=0)

        elif k in test_tuple:
            print('test: {0}'.format(k))
            
            if X_test is None:
                X_test = rgb_imgs_orig
                y_test = labels_orig
                
            else:
                X_test = np.append(X_test,rgb_imgs_orig,axis=0)
                y_test = np.append(y_test,labels_orig,axis=0)

    print()
    return X_train,y_train,X_test,y_test



def generate_beans_leave_one_data_all(dict_orig,dict_aug,key):
    '''generate leave one data with orig and augmentation
    return 4D data, first three as rgb

    '''

    ks = dict_aug.keys()
    rgb_cat = None
    y_train = None
    for k in ks:
        data_aug = dict_aug[k]
        data_orig = dict_orig[k]

        # scale data
        rgb_imgs_aug = data_aug[0]
        rgb_imgs_orig = data_orig[0]
        
        
        labels_aug = data_aug[1]
        labels_orig = data_orig[1]

        if k == key:
            print('test: {0}'.format(k))

            # cat as 4th channel
            X_test = rgb_imgs_orig
            y_test = labels_orig
        else:
            print('train: {0}'.format(k))
            if rgb_cat is None:
                rgb_cat = rgb_imgs_aug
                rgb_cat = np.append(rgb_cat,rgb_imgs_orig,axis=0)

                y_train = labels_aug
                y_train = np.append(y_train,labels_orig,axis=0)
                
            else:
                rgb_cat = np.append(rgb_cat,rgb_imgs_aug,axis=0)
                rgb_cat = np.append(rgb_cat,rgb_imgs_orig,axis=0)

                y_train = np.append(y_train,labels_aug,axis=0)
                y_train = np.append(y_train,labels_orig,axis=0)

    X_train = rgb_cat
    print()
    return X_train,y_train,X_test,y_test



def generate_beans_train_one_data_all(dict_orig,dict_aug,key):
    '''generate leave one data with orig and augmentation
    return 4D data, first three as rgb

    '''

    ks = dict_aug.keys()
    rgb_cat = None
    y_train = None
    for k in ks:
        data_aug = dict_aug[k]
        data_orig = dict_orig[k]

        # scale data
        rgb_imgs_aug = data_aug[0]
        rgb_imgs_orig = data_orig[0]
        
        
        labels_aug = data_aug[1]
        labels_orig = data_orig[1]

        if k == key:
            print('train: {0}'.format(k))

            # cat as 4th channel
            X_train = rgb_imgs_orig
            y_train = labels_orig

            X_train = np.append(X_train, rgb_imgs_aug,axis = 0)
            y_train = np.append(y_train,labels_aug, axis = 0)

        else:
            print('test: {0}'.format(k))
            if rgb_cat is None:
                rgb_cat = rgb_imgs_orig
                y_test = labels_orig
                
            else:
                rgb_cat = np.append(rgb_cat,rgb_imgs_orig,axis=0)
                y_test = np.append(y_test,labels_orig,axis=0)

    X_test = rgb_cat
    print()
    return X_train,y_train,X_test,y_test



def generate_beans_train_two_data_all(dict_orig,dict_aug,keys):
    '''generate leave one data with orig and augmentation
    return 4D data, first three as rgb

    '''

    ks = dict_aug.keys()
    rgb_cat = None
    y_train = None
    X_train = None
    for k in ks:
        data_aug = dict_aug[k]
        data_orig = dict_orig[k]

        # scale data
        rgb_imgs_aug = data_aug[0]
        rgb_imgs_orig = data_orig[0]
        
        
        labels_aug = data_aug[1]
        labels_orig = data_orig[1]

        if k in keys:
            print('train: {0}'.format(k))

            # cat as 4th channel
            if X_train is None:
                X_train = rgb_imgs_orig
                y_train = labels_orig
                
            else:
                X_train = np.append(X_train, rgb_imgs_orig,axis = 0)
                y_train = np.append(y_train,labels_orig, axis = 0)

            X_train = np.append(X_train, rgb_imgs_aug,axis = 0)
            y_train = np.append(y_train,labels_aug, axis = 0)    

        else:
            print('test: {0}'.format(k))
            if rgb_cat is None:
                rgb_cat = rgb_imgs_orig
                y_test = labels_orig
                
            else:
                rgb_cat = np.append(rgb_cat,rgb_imgs_orig,axis=0)
                y_test = np.append(y_test,labels_orig,axis=0)

    X_test = rgb_cat
    print()
    return X_train,y_train,X_test,y_test



def generate_labels(len,num_class):
    assert(len % num_class == 0)

    labels = None
    target_num = int(len / num_class)

    for c in range(num_class):
        tmp = np.ones((target_num,)) * c
        if labels is None:
            labels = tmp
        else:
            labels = np.append(labels,tmp,axis=0)
    return labels



def generate_regression_labels(len):
    labels = []

    for i in range(len):
        labels.append(i + 1)

    labels = np.array(labels)
    labels = labels / len

    return labels



def load_imgs_labels(folder):

    rgb_folder = join(folder,'rgb')
    rgb_imgs = load_imgs(rgb_folder)

    flir_folder = join(folder,'flir')
    flir_imgs = load_imgs(flir_folder)

    # load labels
    labels = np.loadtxt(join(folder,'labels.txt'))

    # load avg Ts
    file = join(folder,'avg_Ts.txt')
    ts = None
    if isfile(file):
        ts = np.loadtxt(file)

    assert(labels.shape[0] == flir_imgs.shape[0])

    return flir_imgs,rgb_imgs,labels,ts



def load_imgs_from_folders(folder,load_aug,issubtract,isregression=False):
    '''load imgs from BOTH original and augment folder or JUST original folder
    '''

    lists = listdir(folder)
    lists = natsorted(lists)

    dict_aug = None
    if load_aug:
        dict_aug = {}
    dict_orig = {}
    for f in lists:
        # use capture num and t-setting as key
        idx = f.find('-')
        idx2 = f.rfind('-')
        capture_num = f[0:idx2]
        print('folder name: {0}'.format(f))
        print('key: {0}'.format(capture_num))


        subf = join(folder,f)
        flir_imgs2,rgb_imgs2,labels2,ts2 = load_imgs_labels(subf)
        if isregression:
            labels2 = generate_regression_labels(labels2.shape[0])

        if issubtract:
            # truncate to same length
            ts2 = ts2[0:flir_imgs2.shape[0]]
            for i in range(len(flir_imgs2.shape) - 1):
                ts2 = ts2[...,np.newaxis]
            ts2 = np.tile(ts2,(1,flir_imgs2.shape[1],flir_imgs2.shape[2],flir_imgs2.shape[3]))
            flir_imgs2 = np.absolute(ts2 - flir_imgs2)

        dict_orig[capture_num] = (flir_imgs2,rgb_imgs2,labels2)


        if load_aug:
            subf = join(folder,f,'train')
            flir_imgs,rgb_imgs,labels,ts = load_imgs_labels(subf)

            assert(labels.shape[0] % labels2.shape[0] == 0)
            aug_size = int(labels.shape[0] / labels2.shape[0])

            if isregression:
                labels = np.repeat(labels2,aug_size,axis=0)

            if issubtract:
                ts_aug = np.repeat(ts2,aug_size,axis=0)
                flir_imgs = np.absolute(flir_imgs - ts_aug)

            dict_aug[capture_num] = (flir_imgs,rgb_imgs,labels)
            
    return dict_orig,dict_aug




def generate_n_fold_data(n,dict_orig,dict_aug):
    keys = list(dict_orig.keys())
    assert(len(keys) % n == 0)

    if dict_aug is None:
        load_aug = False
    else:
        load_aug = True

    target_num = len(keys) / n

    SEED = 7
    seed(SEED)

    shuffle(keys)

    print('\nrandom shuffle...\n')
    print('\n'.join(keys))
    print('')

    dict_nfold_aug = None
    if load_aug:
        dict_nfold_aug = {}
    dict_nfold_orig = {}
    

    flir_cat = None
    rgb_cat = None
    y_cat = None

    flir_orig_cat = None
    rgb_orig_cat = None
    y_orig_cat = None

    
    for i,k in enumerate(keys,1):
        print('cat {0}'.format(k))

        if load_aug:
            data = dict_aug[k]
            flir_imgs = data[0]
            rgb_imgs = data[1]
            labels = data[2]

        data_orig = dict_orig[k]
        flir_imgs_orig = data_orig[0]
        rgb_imgs_orig = data_orig[1]
        labels_orig = data_orig[2]

        # concat data
        if flir_orig_cat is None:
            if load_aug:
                flir_cat = flir_imgs
                rgb_cat = rgb_imgs
                y_cat = labels

            flir_orig_cat = flir_imgs_orig
            rgb_orig_cat = rgb_imgs_orig
            y_orig_cat = labels_orig
        else:
            if load_aug:
                flir_cat = np.append(flir_cat,flir_imgs,axis=0)
                rgb_cat = np.append(rgb_cat,rgb_imgs,axis=0)
                y_cat = np.append(y_cat,labels,axis=0)

            flir_orig_cat = np.append(flir_orig_cat,flir_imgs_orig,axis=0)
            rgb_orig_cat = np.append(rgb_orig_cat,rgb_imgs_orig,axis=0)
            y_orig_cat = np.append(y_orig_cat,labels_orig,axis=0)
            

        if i % target_num == 0:
            print('split here\n')
            newk = int(i / target_num)
            if load_aug:
                dict_nfold_aug[newk] = (flir_cat,rgb_cat,y_cat)
            dict_nfold_orig[newk] = (flir_orig_cat,rgb_orig_cat,y_orig_cat)

            flir_cat = None
            rgb_cat = None
            y_cat = None
            flir_orig_cat = None
            rgb_orig_cat = None
            y_orig_cat = None


    return dict_nfold_orig,dict_nfold_aug




def generate_leave_one_data_all(dict_orig,dict_aug,key):
    '''generate leave one data with orig and augmentation
    return 4D data, first three as rgb, last one as flir

    '''

    ks = dict_aug.keys()
    flir_cat = None
    rgb_cat = None
    y_train = None
    for k in ks:
        data = dict_aug[k]
        data_orig = dict_orig[k]

        # scale data
        flir_imgs = data[0]
        flir_imgs_orig = data_orig[0]
        
        rgb_imgs = data[1] 
        rgb_imgs_orig = data_orig[1] 
        
        labels = data[2]
        labels_orig = data_orig[2]

        if k == key:
            print('test: {0}'.format(k))

            # cat as 4th channel
            X_test = np.append(rgb_imgs_orig,flir_imgs_orig,axis=-1)
            y_test = labels_orig
        else:
            print('train: {0}'.format(k))
            if flir_cat is None:
                flir_cat = flir_imgs
                flir_cat = np.append(flir_cat,flir_imgs_orig,axis=0)

                y_train = labels
                y_train = np.append(y_train,labels_orig,axis=0)
                
                rgb_cat = rgb_imgs
                rgb_cat = np.append(rgb_cat,rgb_imgs_orig,axis=0)
            else:
                flir_cat = np.append(flir_cat,flir_imgs,axis=0)
                flir_cat = np.append(flir_cat,flir_imgs_orig,axis=0)

                y_train = np.append(y_train,labels,axis=0)
                y_train = np.append(y_train,labels_orig,axis=0)

                rgb_cat = np.append(rgb_cat,rgb_imgs,axis=0)
                rgb_cat = np.append(rgb_cat,rgb_imgs_orig,axis=0)

    # cat as 4th channel
    X_train = np.append(rgb_cat,flir_cat,axis=-1)

    return X_train,y_train,X_test,y_test



def generate_leave_one_data_orig(dict,key):
    '''generate leave one data with ONLY original data
    return 4D data, first three as rgb, last one as flir
    '''
    ks = dict.keys()
    flir_cat = None
    rgb_cat = None
    y_train = None
    for k in ks:
        data = dict[k]
        flir_imgs = data[0] 
        rgb_imgs = data[1] 
        labels = data[2]

        if k == key:
            print('test: {0}'.format(k))

            # cat as 4th channel
            X_test = np.append(rgb_imgs,flir_imgs,axis=-1)
            y_test = labels
        else:
            print('train: {0}'.format(k))
            if flir_cat is None:
                flir_cat = flir_imgs
                rgb_cat = rgb_imgs
                y_train = labels
            else:
                flir_cat = np.append(flir_cat,flir_imgs,axis=0)
                rgb_cat = np.append(rgb_cat,rgb_imgs,axis=0)
                y_train = np.append(y_train,labels,axis=0)

    # cat as 4th channel
    X_train = np.append(rgb_cat,flir_cat,axis=-1)

    return X_train,y_train,X_test,y_test



def rollback(data_4d):
    '''4D data roll back to rgb and flir
    '''
    rgb = data_4d[:,:,:,0:3]

    flir = data_4d[:,:,:,3]
    flir = flir[...,np.newaxis]

    return rgb,flir



def scale_imarray(array,nbits=8):
    scaled_array = array / (2 ** nbits - 1)
    return scaled_array



def save_imarray(array,folder,labels,predicts,precision=2,map='jet'):

    assert(len(array.shape) == 4)

    is_flir = False
    if array.shape[-1] == 1:
        is_flir = True
        newarr = np.squeeze(array,axis=-1)
    elif array.shape[-1] > 3:
        newarr = array[:,:,:,:3]
    else:
        newarr = array
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    if is_flir:
        im = ax.imshow(newarr[0],cmap=map)
    else:
        im = ax.imshow(newarr[0])
    for i in range(array.shape[0]):

        im.set_data(newarr[i])
        fname = join(folder,str(i) + '.tiff')

        l1 = labels[i].astype(float)
        p1 = predicts[i].astype(float)
        ax.set_title('label: {0}\n predict: {1}'.format(np.around(l1,precision), np.around(p1,precision))) 
        fig.savefig(fname)



def top_2_accuracy(y_true,y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,2)



def calculate_top_k_acc(labels,predicts,top_k):
    '''calculate adjacent top k accuracy
    '''

    assert(len(labels.shape) == 1)
    assert(len(predicts.shape) == 2)

    predicts = np.argpartition(predicts,-top_k)[:,-top_k:]

    count = 0
    for i in range(predicts.shape[0]):
        if labels[i] in predicts[i]:
            sorted_predicts = np.sort(predicts[i])
            l2 = np.array(range(0,top_k)) + sorted_predicts[0]
            tmp = sum(sorted_predicts == l2)
            if tmp == predicts[i].shape[0]:
                count +=1             

    acc = count / predicts.shape[0]
    return acc

    '''
    labels=labels[...,np.newaxis]
    labels=np.repeat(labels,top_k,axis=-1)
    log=predicts==labels
    top_k_acc=np.zeros(log[:,0].shape)
    for i in range(log.shape[-1]-1):
        tmp = np.logical_or(log[:,i],log[:,i+1])
        top_k_acc = np.logical_or(top_k_acc,tmp)

    return np.mean(top_k_acc)'''


def get_cv_LSTM(X_train, Y_train, video_length, total_videos, part):
    SEED = 7
    seed(SEED)
    field = list(range(0,total_videos-part))
    shuffle(field)
    indx = field[0:part] 
    dX_train = None
    dY_train = None
    X_cv = np.copy(X_train[indx[0]*video_length:(indx[0]+1)*video_length])
    
    Y_cv = np.copy(Y_train[indx[0]*video_length:(indx[0]+1)*video_length])
    cv_list = list(range(indx[0]*video_length,(indx[0]+1)*video_length))
    for i in range(1,part):
        X_cv = np.append(X_cv, np.copy(X_train[indx[i]*video_length:(indx[i]+1)*video_length]), axis = 0)
        Y_cv = np.append(Y_cv, np.copy(Y_train[indx[i]*video_length:(indx[i]+1)*video_length]), axis = 0)
        cv_list += list(range(indx[i]*video_length,(indx[i]+1)*video_length))
   
    for i in range(0, X_train.shape[0]):
        if i in cv_list:
            pass
        else:
            if dX_train is None:
                dX_train = np.copy(X_train[i])
            else:
                dX_train = np.append(dX_train, X_train[i], axis = 0)
    img_shape = X_train[0].shape
    dX_train = dX_train.reshape(dX_train.shape[0]//img_shape[0],img_shape[0], img_shape[1], img_shape[2])
    for i in range(0, X_train.shape[0]):
        if i in cv_list:
            pass
        else:
            if dY_train is None:
                dY_train = [Y_train[i]]
            else:
                dY_train.append(Y_train[i])
    dY_train = np.array(dY_train)

    return dX_train, dY_train, X_cv, Y_cv
 

def regroup_train_LSTM(train_data, samples):

    shape = train_data.shape
    new_train = []
    
    for j in range(0, samples):
        for i in range(0, shape[0]//samples):
            new_train.append(train_data[i*samples+j])
    new_train = np.array(new_train)
    return new_train


def split_train_cv_LSTM(X_train_index, Y_train_index, video_length, part, validation_ratio = 0.2):
    SEED = 7
    seed(SEED)

    X_index, Xcv_index,Y_index, Y_cv_index = train_test_split(X_train_index, Y_train_index, test_size = 0.2)
    return X_index, Xcv_index,Y_index, Y_cv_index

def load_KLT_data(folder):
    files = listdir(folder)
    files = natsorted(files)
    data = []
    for f in files:
        datafile = join(folder,  f)
        img = sio.loadmat(datafile)
        #img = Image.open(datafile)
        #imgarray = np.array(img,dtype=np.uint16)
        img_np = np.array(img['KLTload'], dtype=np.float)
        data.append(img_np)
    data = np.array(data)
    return data

def load_KLT_from_folders(folder):
    dict_KLT_orig = {}
    lists = listdir(folder)
    lists = natsorted(lists)
    for f in lists:
        KLTfile = join(folder, f, 'KLTtrackingtrainkltmat')
        indx = f.find('-')
        indx2 = f.rfind('-')
        capnum = f[0:indx2]
        mat_data = load_KLT_data(KLTfile)
        dict_KLT_orig[capnum] = mat_data
    return dict_KLT_orig

def generate_n_folder_KLT(n, dict_KLT_orig):
    keys = list(dict_KLT_orig.keys())
    dict_orig = {}
    assert(len(keys) % n == 0)
    target_num = len(keys) / n

    SEED = 7
    seed(SEED)

    shuffle(keys)

    datamat = []
    for i, k in enumerate(keys, 1):
        datamat.append(dict_KLT_orig[k])
        if i%target_num == 0:
            newkey = int(i/target_num)
            dict_orig[newkey] = np.array(datamat)
            datamat = []

    return dict_orig

def generate_leave_one_KLTdata_orig(dict,key):
   ks = list(dict.keys())
   X_train = None
   for k in ks:
       if k==key:
           X_test = dict[k]
           shape = X_test.shape
           X_real_test = None
           for i in range(0, shape[0]):
               if X_real_test is None:
                   X_real_test = X_test[0]
               else:
                   X_real_test = np.append(X_real_test, X_test[i], axis = 0)
           y_tmp = list(range(1, shape[1]+1))*shape[0]
           Y_test = np.array(y_tmp)/shape[1]
       else:
           if X_train is None:
               X_train = dict[k]
           else:
               X_train = np.append(X_train, dict[k], axis = 0)
   X_real_train = None
   for i in range(0, X_train.shape[0]):
       if X_real_train is None:
           X_real_train = X_train[i]
       else:
           X_real_train = np.append(X_real_train, X_train[i], axis = 0)
   video_len = dict[ks[0]].shape[1]
   repeat = X_train.shape[0]
   y_tmp = list(range(1, video_len+1))*repeat
   Y_train = np.array(y_tmp)/video_len
   return X_real_train, Y_train, X_real_test, Y_test

def generate_combined_data(X_train, Y_train, X_KLT_train, Y_KLT_train, X_test, Y_test, X_KLT_test, Y_KLT_test):
    X_combined_train = []
    Y_combined_train = []
    X_combined_test = []
    Y_combined_test = []
    
    for i in range(0, X_train.shape[0]):
        X_combined_train.append([X_train[i], X_KLT_train[i]])
    for i in range(0, X_test.shape[0]):
        X_combined_test.append([X_test[i], X_KLT_test[i]])
    Y_combined_train = Y_train
    Y_combined_test = Y_test
    return X_combined_train, Y_combined_train, X_combined_test, Y_combined_test