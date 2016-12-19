# coding: utf-8

# In[ ]:

#caltech101.py
import os
import glob
import cv2
from sklearn.utils import shuffle
from scipy.misc import imread
from keras.utils import np_utils
import numpy as np

tr_dir="/data/global/gaurav/caltech101/101_ObjectCategories"

h,w,c,d,=64,64,102,3

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
    return resized

def load_data():
    X_train = []
    X_train_id = []
    y_train = []
    files_all=[]
    index=0
    
    for fl in os.listdir(tr_dir):
        fld=os.path.join(tr_dir,fl)
        print('load folder {} (Index:{})'.format(fld, index))
        path = os.path.join(fld, '*.jpg')
        files = sorted(glob.glob(path))
        for fl in files:
            flbase = os.path.basename(fl)
            files_all.append(fl)
            X_train_id.append(flbase)
            y_train.append(index)
                     
        index = index+1
    
    files_all, y_train= shuffle(files_all, y_train, random_state=9999)
    files_all, y_train= shuffle(files_all, y_train, random_state=9999)
    print("SHUFFLED TRAINING FILES")

    for fl in files_all:
        img = get_im_cv2(fl)
        X_train.append(img)

    print("Collected data TRAINING FILES")

    X_train=np.array(X_train,dtype=np.uint8)
    y_train=np.array(y_train,dtype=np.uint8)
    X_train=X_train.transpose((0,3, 1, 2))
    X_train=X_train.astype('float32')
    X_train=X_train/255    
    y_flat=y_train
    y_train=np_utils.to_categorical(y_train,c)
    print("PREPARED TRAINING DATA")
    from sklearn.model_selection import StratifiedShuffleSplit as ss
    sss = ss(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in sss.split(X_train,y_train):
        x_t,x_v=X_train[train_index],X_train[test_index]
        y_t,y_v=y_train[train_index],y_train[test_index]
        y_f,y_f_v=y_flat[train_index],y_flat[test_index]
        
    return x_t,y_t,x_v,y_v,y_f
#caltech101.py
