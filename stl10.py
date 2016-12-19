#stl10.py
import stl10_input
import numpy as np
from keras.utils import np_utils

#print(stl10_input.TR_DATA_PATH)
#print(stl10_input.TS_DATA_PATH)
c=10

def load_data():
    x_t,y_f,x_v,y_tvf=stl10_input.read_stl10_data()
    x_t=np.array(x_t,dtype=np.uint8)
    y_f=np.array(y_f,dtype=np.uint8)
    x_t=x_t.transpose((0,3, 1, 2))
    x_t=x_t.astype('float32')
    x_t=x_t/255    
    y_t=np_utils.to_categorical(y_f,c)

    #TEST DATA
    x_v=np.array(x_v,dtype=np.uint8)
    y_v=np.array(y_tvf,dtype=np.uint8)
    x_v=x_v.transpose((0,3, 1, 2))
    x_v=x_v.astype('float32')
    x_v=x_v/255    
    y_v=np_utils.to_categorical(y_v,c)
    print x_t.shape
    print x_v.shape
    return x_t,y_t,x_v,y_v,y_f
#stl10.py
