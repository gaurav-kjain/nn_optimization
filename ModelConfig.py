#ModelConfig.py
from dnn_train import *
from gsnet import *
import sys
from dnn_globals import DirGlobals
#from IPython.display import clear_output
import os
import pandas as pd
#import stl10
try:
    import cPickle as pickle
except:
    import pickle

sys.setrecursionlimit(600000)
print("SETTING RECURSION LIMIT")
epoc__ = 1000
ext__='.pkk'
mext__='.mdd'
gloss='categorical_crossentropy'

def makefinresults_filename(b, e, opti, ext):
    finstr = str(b) + 'B' + '_' + str(e) + 'E' + '_'
    finstr = finstr + opti
    finstr = finstr + ext
    return finstr


def makefinresults_filename_init(b, e, opti, init, ext):
    finstr = str(b) + 'B' + '_' + str(e) + 'E' + '_'
    finstr = finstr + opti + '_' + init
    finstr = finstr + ext
    return finstr


def getdirectorypath(main, db, nwdir):
    directory = os.path.join(main, db)
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(directory, nwdir)
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def getfinresname(direc, filenm):
    directory = os.path.join(direc, filenm)
    return (os.path.abspath(directory))


def printconfig(nw, db, opti, bsize):
    print(db, opti, bsize)
    print(nw())


def save_results(hist, finfilename):
    acc         = hist.history['acc']
    val_acc     = hist.history['val_acc']
    loss        = hist.history['loss']
    val_loss    = hist.history['val_loss']
    np.savez(finfilename, acc=acc, val_acc=val_acc, loss=loss, val_loss=val_loss)


def loadResults(db, nw, filename):
    dp = getdirectorypath(MAIN_DIR, dbDict[db], nwDict[nw])
    finfilename = getfinresname(dp, filename)
    hist = np.load(finfilename)
    return hist

def loadData(datastring='mnist', cnnin=True):

    img_h, img_w, d, nb_classes = 32, 32, 1, 10
    X_train, Y_train, X_test, Y_test =[],[],[],[]

    if datastring is 'mnist':
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        img_h, img_w, d, nb_classes = 28, 28, 1, 10

    elif datastring is 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        img_h, img_w, d, nb_classes = 32, 32, 3, 10

    elif datastring is 'cifar100':
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')
        img_h, img_w, d, nb_classes = 32, 32, 3, 100

    elif datastring is 'cifar20':
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='coarse')
        img_h, img_w,d,nb_classes = 32, 32, 3, 20

    elif datastring is 'flowers_102':
        #x_t,y_t,x_v,y_v,y_f = flowers101.load_data()        
        img_h, img_w, d, nb_classes = 128, 128, 3, 102
        return img_h,img_w,d,nb_classes,x_t,y_t,x_v,y_v,y_f


    elif datastring is 'caltech_101':
        #x_t,y_t,x_v,y_v,y_f = caltech101.load_data()        
        img_h, img_w, d, nb_classes = 64, 64, 3, 102
        return img_h,img_w,d,nb_classes,x_t,y_t,x_v,y_v,y_f

    elif datastring is 'stl_10':
        #X_train, Y_train, X_test, Y_test,Y_flat = stl10.load_data()
        img_h, img_w, d, nb_classes = 96, 96, 3, 10
        return img_h, img_w, d, nb_classes, X_train, Y_train, X_test, Y_test, Y_flat

    Y_flat = Y_train

    if cnnin is True:
        X_train = X_train.reshape(X_train.shape[0], d, img_h, img_w)
        X_test = X_test.reshape(X_test.shape[0], d, img_h, img_w)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    return img_h, img_w, d, nb_classes, X_train, Y_train, X_test, Y_test, Y_flat


TWOLAYER_DIR = 'TwoLayer'
TWOLAYER_N_DIR = 'TwoLayerNoise'
TWOLAYER_H_DIR = 'TwoLayerHier'

THREELAYER_DIR = 'ThreeLayer'
THREELAYER_N_DIR = 'ThreeLayerNoise'
THREELAYER_H_DIR = 'ThreeLayerHier'

FIVELAYER_DIR = 'FiveLayer'
FIVELAYER_N_DIR = 'FiveLayerNoise'
FIVELAYER_H_DIR = 'FiveLayerHier'

MERGE_WEAK_2_DIR = 'MergeWeak2'
MERGE_WEAK_3_DIR = 'MergeWeak3'
GSNET2_DIR='gsnet2'
GSNET3_DIR='gsnet3'
GSNET5_DIR='gsnet5'
####PARAM DICTIONARY

dbName = {
    'MNIST'     : 'mnist',
    'CIFAR10'   : 'cifar10',
    'CIFAR100'  : 'cifar100',
    'CIFAR20'   : 'cifar20',
    'FLOWERS102': 'flowers_102',
    'CALTECH101': 'caltech_101',
    'STL10'     : 'stl_10'
}

dbcsvName = {
    'MNIST'     : 'mnist',
    'CIFAR10'   : 'cif10',
    'CIFAR100'  : 'cif100',
    'CIFAR20'   : 'cif20',
    'FLOWERS102': 'fl102',
    'CALTECH101': 'cal101',
    'STL10'     : 'stl10'
}

configName = {
    'normal'            : 'normal',
    'careful'           : 'careful',
    #'careful_init'      : 'careful_init',
    #'outliers_iqr_npass': 'outliers_iqr_npass',
    #'outliers_iqr'      : 'outliers_iqr'
}

nwDict = {
    #'GSNET_2'       :  GSNET2_DIR,
    #'GSNET_3'       :  GSNET3_DIR,
    #'GSNET_5'       :  GSNET5_DIR,
    'TWOLAYER'     :  TWOLAYER_DIR,
    #'TWOLAYER_N'   :  TWOLAYER_N_DIR,
    'TWOLAYER_H'   :  TWOLAYER_H_DIR,

    'THREELAYER'   :  THREELAYER_DIR,
    #'THREELAYER_N' :  THREELAYER_N_DIR,
    'THREELAYER_H' :  THREELAYER_H_DIR,

    'FIVELAYER'    :  FIVELAYER_DIR,
    #'FIVELAYER_N'  :  FIVELAYER_N_DIR,
    'FIVELAYER_H'  :  FIVELAYER_H_DIR,
    #'MERGE_WEAK_2' :  MERGE_WEAK_2_DIR,
    #'MERGE_WEAK_3' :  MERGE_WEAK_3_DIR
}

nwcsvName = {
    #'GSNET_2'       :  '',
    #'GSNET_3'       :  GSNET3_DIR,
    #'GSNET_5'       :  GSNET5_DIR,
    'TWOLAYER'       :  'twl',
    #'TWOLAYER_N'   :  TWOLAYER_N_DIR,
    #'TWOLAYER_H'   :  TWOLAYER_H_DIR,

    'THREELAYER'     : 'thl',
    #'THREELAYER_N' :  THREELAYER_N_DIR,
    #'THREELAYER_H' :  THREELAYER_H_DIR,

    'FIVELAYER'      : 'fvl',
    #'FIVELAYER_N'  :  FIVELAYER_N_DIR,
    #'FIVELAYER_H'  :  FIVELAYER_H_DIR,
    #'MERGE_WEAK_2' :  MERGE_WEAK_2_DIR,
    #'MERGE_WEAK_3' :  MERGE_WEAK_3_DIR
}

nwFunc = {
    'GSNET_2'       :  gsnet_cnn_2,
    'GSNET_3'       :  gsnet_cnn_3,
    'GSNET_5'       :  gsnet_cnn_5,
    'TWOLAYER': two_layer_Model,
    'TWOLAYER_N': two_layer_Model_Noisy,
    'TWOLAYER_H': two_layer_Model_hier,
    'THREELAYER_H': three_layer_Model_hier,
    'THREELAYER': three_layer_Model,
    'THREELAYER_N': three_layer_Model_Noisy,
    'FIVELAYER': five_layer_Model,
    'FIVELAYER_N': five_layer_Model_Noisy,
    'FIVELAYER_H': five_layer_Model_Hier,
    'MERGE_WEAK_2': merge_weak_layer_Models,
    'MERGE_WEAK_3': merge_wide3_layer_Models,
}
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

optimizerDict = {
    'ADL': 'adadelta',
    'ADG': 'adagrad',
    'SGD':  sgd,
    'ADM': 'adam',
    'ADMX': 'adamax',
    'NDM' :'nadam',
}

initDict = {
    'unif': 'uniform',
    'norm': 'normal',
    'zero': 'zero',
    'gl_n': 'glorot_normal',
    'gl_u': 'glorot_uniform',
    'he_u': 'he_uniform',
}

print(optimizerDict)

def save_results_via_pickle(hist, finfilename):
    with open(finfilename, 'wb') as histarr:
        pickle.dump(hist, histarr, pickle.HIGHEST_PROTOCOL)
    
def readpickle(picklefile):
    data=[]
    with open(picklefile, 'rb') as ppf:
        data=pickle.load(ppf)
    return data       

def makecsv_careful_init(picklefile):
    data=readpickle(picklefile)
    df=pd.DataFrame()
    df1=pd.DataFrame()
    
    for idx in range(0,4,1):
        df[data[idx][0]]=data[idx][1]
    for idx in range(4,8,1):
        df1[data[idx][0]]=data[idx][1]
    df=pd.concat([df,df1]).reset_index()
    return df
 
def makecsv_carenormal(picklefile):
    data=readpickle(picklefile)
    df=pd.DataFrame()
    for idx in range(0,4,1):
        df[data[idx][0]]=data[idx][1]
    df=df.reset_index()
    return df

def makecsv_outliers(picklefile):
    data=readpickle(picklefile)
    df=pd.DataFrame()
    offset=20
    tot=len(data)/offset
    peroff=4
    
    for outidx in range(0,tot,1):
        df1=pd.DataFrame()
        for idx in range(0,peroff,1):
            df1[data[outidx*offset+idx][0]]=data[outidx*offset+idx][1]
        for idx in range(peroff,offset,2):
            df1[data[outidx*offset+idx]]=data[outidx*offset+idx+1]
        df=pd.concat([df,df1])
    df=df.reset_index()
    return df

def makecsv_outliers_npass(picklefile):
    data=readpickle(picklefile)
    df=pd.DataFrame()
    df1=pd.DataFrame()
    offset=20
    tot=len(data)/offset
    peroff=4
    
    for outidx in range(0,tot,1):
        df_=pd.DataFrame()
        for idx in range(0,peroff,1):
            df_[data[outidx*offset+idx][0]]=data[outidx*offset+idx][1]
        for idx in range(peroff,offset,2):
            df_[data[outidx*offset+idx]]=data[outidx*offset+idx+1]
        df=pd.concat([df,df1])
    df=df.reset_index()
    return df

def makecsv_global_write(picklefile,towritefile, descfile, bmode):

    towritedata=[]
    if bmode is 'normal':
        towritedata=makecsv_carenormal(picklefile)
    elif bmode is 'careful':
        towritedata=makecsv_carenormal(picklefile)
    elif bmode is 'careful_init':
        towritedata=makecsv_careful_init(picklefile)
    elif bmode is 'outliers_iqr':
        towritedata=makecsv_outliers(picklefile)
    elif bmode is 'outliers_iqr_npass':
        towritedata=makecsv_outliers(picklefile)

    towritedata.to_csv(towritefile)
    desdata=towritedata.describe()
    desdata.to_csv(descfile)
    return desdata




def makecsv_global_write_npz(npzfile,towritefile, descfile, bmode):

    npdata=np.load(npzfile)
    towritedata=pd.DataFrame()
    towritedata['acc']=npdata['acc']
    towritedata['val_acc']=npdata['val_acc']
    towritedata['loss']=npdata['loss']
    towritedata['val_loss']=npdata['val_loss']
    print(towritedata)
    towritedata.to_csv(towritefile)
    desdata=towritedata.describe()
    desdata.to_csv(descfile)
    return desdata


def makedf_global_from_npz(npzfile):

    npdata=np.load(npzfile)
    towritedata=pd.DataFrame()
    towritedata['acc']=npdata['acc']
    towritedata['val_acc']=npdata['val_acc']
    towritedata['loss']=npdata['loss']
    towritedata['val_loss']=npdata['val_loss']
    print(towritedata)
    return towritedata
