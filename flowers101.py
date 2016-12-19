
#flowers101.py

import numpy as np
import os
import cv2
from sklearn.utils import shuffle
from scipy.misc import imread
import numpy as np
from keras.utils import np_utils

train_path="/data/global/gaurav/flowers102/caffe-oxford102/train.txt"
test_path="/data/global/gaurav/flowers102/caffe-oxford102/test.txt"

lines_tr = tuple(open(train_path, 'r'))
lines_ts = tuple(open(test_path, 'r'))
h,w,c,d=128,128,102,3
def get_im_cv2(path):
    img = cv2.imread(path)
    #if img:
    img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
    return img

def get_data(lines):
        X_train = []
        X_train_id = []
        y_train = []
        files_all=[]
        index=0
        for fld in lines:
            path,index=fld.split(' ')
            index=int(index)
            #print('load folder {} (Index:{})'.format(fld, index))
            files_all.append(path)
            y_train.append(index)
        
        print(len(files_all))
        files_all, y_train= shuffle(files_all, y_train, random_state=9999)
        files_all, y_train= shuffle(files_all, y_train, random_state=9999)
        print("SHUFFLED TRAINING FILES")

        for fl in files_all:
            #print(fl)
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
        return X_train,y_train,y_flat


def load_data():
    x_t,y_t,y_f= get_data(lines_tr)
    x_v,y_v,y_tf= get_data(lines_ts)
    print('COLLECTED VALUES')
    return x_t,y_t,x_v,y_v,y_f

#flowers101.py

