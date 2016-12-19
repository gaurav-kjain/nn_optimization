#dnn_thesis_u.py
import os

import numpy as np
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Activation, Merge
from keras.optimizers import SGD
import time
import stl10
import caltech101
import flowers101

print('FROM DNN_OPTI DIRECTORY')
BATCH_MODE=set(['normal','careful','up','carefulup','outliers'])



nb_filters = 64
nb_pool = 2
batsiz_multiplier = 2
ext__ = '.npz'
epoc__ = 1000
init__ = 'lecun_uniform'
patience__ = 50
gloss = 'categorical_crossentropy'


def merge_wide3_layer_Models(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                             classes=10, init='adam'):
    left = Sequential()
    left.add(GaussianNoise(sigma=0.01, input_shape=(depth, img_rows, img_cols)))
    left.add(Convolution2D(classes * 4, 3, 3, border_mode='valid'))

    left.add(Activation('relu'))
    left.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    left.add(Convolution2D(nb_filters * 2, 3, 3))
    left.add(Activation('relu'))
    left.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    left.add(Dropout(0.25))
    left.add(Flatten())
    left.add(Dense(classes * 10))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(GaussianNoise(sigma=0.2, input_shape=(depth, img_rows, img_cols)))
    right.add(Convolution2D(classes * 4, 3, 3, border_mode='valid'))

    right.add(Activation('relu'))
    right.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    right.add(Convolution2D(nb_filters, 3, 3))
    right.add(Activation('relu'))
    right.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    right.add(Dropout(0.25))
    right.add(Flatten())
    right.add(Dense(classes * 10))
    right.add(Activation('relu'))

    middle = Sequential()
    middle.add(Convolution2D(classes * 4, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    middle.add(Activation('relu'))
    middle.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    middle.add(Convolution2D(nb_filters, 3, 3))
    middle.add(Activation('relu'))
    middle.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    middle.add(Dropout(0.25))
    middle.add(Flatten())
    middle.add(Dense(classes * 10))
    middle.add(Activation('relu'))

    merged = Merge([left, middle, right], mode='concat')

    final = Sequential()
    final.add(merged)
    final.add(Dropout(0.5))
    final.add(Dense(classes))
    final.add(Activation('softmax'))
    final.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return final


def merge_weak_layer_Models(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                            classes=10, init='adam'):
    left = Sequential()
    left.add(GaussianNoise(sigma=0.01, input_shape=(depth, img_rows, img_cols)))
    left.add(Convolution2D(classes * 4, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    left.add(Activation('relu'))
    left.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    left.add(Convolution2D(nb_filters * 2, 3, 3))
    left.add(Activation('relu'))
    left.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    left.add(Dropout(0.25))
    left.add(Flatten())
    left.add(Dense(classes * 10))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Convolution2D(classes * 4, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    right.add(Activation('relu'))
    right.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    right.add(Convolution2D(nb_filters, 3, 3))
    right.add(Activation('relu'))
    right.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    right.add(Dropout(0.25))
    right.add(Flatten())
    right.add(Dense(classes * 10))
    right.add(Activation('relu'))

    merged = Merge([left, right], mode='concat')

    final = Sequential()
    final.add(merged)
    final.add(Dropout(0.5))
    final.add(Dense(classes))
    final.add(Activation('softmax'))
    final.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return final


def two_layer_Model_hier(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                         classes=10, init='adam'):
    model = Sequential()
    model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def three_layer_Model_hier(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                           classes=10, init='adam'):
    model = Sequential()
    model.add(Convolution2D(nb_filters * 3, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters * 2, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def two_layer_Model_Noisy(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                          classes=10, init='adam'):
    model = Sequential()
    model.add(GaussianNoise(sigma=0.01, input_shape=(depth, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def two_layer_Model(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                    classes=10, init='adam'):
    model = Sequential()
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def three_layer_Model_Noisy(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                            classes=10, init='adam'):
    model = Sequential()
    model.add(GaussianNoise(sigma=0.01, input_shape=(depth, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def three_layer_Model(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                      classes=10, init='adam'):
    model = Sequential()
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def five_layer_Model_Noisy(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                           classes=10, init='adam'):
    model = Sequential()
    model.add(GaussianNoise(sigma=0.01, input_shape=(depth, img_rows, img_cols)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def five_layer_Model(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                     classes=10, init='adam'):
    model = Sequential()

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def five_layer_Model_Hier(loss='categorical_crossentropy', optimizer='adadelta', img_rows=28, img_cols=28, depth=1,
                          classes=10, init='adam'):
    model = Sequential()

    model.add(Convolution2D(nb_filters * 5, 3, 3, border_mode='valid', input_shape=(depth, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters * 3, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def fitModel_(model, X_train, Y_train, X_test, Y_test, y_f, batch_size=100, nb_epoch=100, verbose=0, c=10, bmode='normal'):
    totepoch=20
    l_th=1e-02
    history = History()
    stopearly = EarlyStopping(monitor='val_acc', patience=patience__, verbose=1, mode='auto')
    #stopearly = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
    time.sleep(0.1)
    if bmode is 'normal':
        print('running without generator')
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=verbose, callbacks=[history,stopearly],validation_data=(X_test, Y_test) )
    elif bmode is 'careful':
        print('running careful generator')
        newgen = generator_of_new(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                            validation_data=(X_test, Y_test), samples_per_epoch=X_train.shape[0])
    elif bmode is 'careful_init':
        print('running careful init generator')
        newgen = generator_of_new(X_train, y_f, c, batch_size)
        stopearly1 = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly1],
                            validation_data=(X_test, Y_test), samples_per_epoch=X_train.shape[0])
        hist1=history.history
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=verbose, callbacks=[history,stopearly],validation_data=(X_test, Y_test) )
        #print(hist1)
        #print(history.history)
        #history=np.vstack( (hist1, history.history) )
        history=hist1.items()+history.history.items()
        print('Done with careful init run')
    elif bmode is 'up':
        print('running up normal generator')
        newgen = generator_of_new(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                            validation_data=(X_test, Y_test), samples_per_epoch=X_train.shape[0])
    elif bmode is 'carefulup':
        print('running up careful generator')
        newgen = generator_of_new(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                            validation_data=(X_test, Y_test), samples_per_epoch=X_train.shape[0])
    
    elif bmode is 'outliers':
        print('running outliers generator')
        x_in=X_train
        y_in=Y_train
        hist11=[]
        stopearly1 = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
        for epp in range(1,totepoch,1):
            print("Epoch="+str(epp)+"/"+str(totepoch))
            hist1=model.fit(x_in, y_in, batch_size=batch_size, nb_epoch=10,verbose=verbose, callbacks=[stopearly1],validation_data=(X_test, Y_test) )
            hist11=hist11+hist1.history.items()+list([x_in.shape[0]])
#            print(hist11)
            probs=model.predict_proba(X_train, batch_size=batch_size,verbose=verbose)
            cross_entr=Y_train*np.log(probs)
            loss_arr=-np.sum(cross_entr,axis=1)#log loss, cross entropy error
            loss_idx=np.where(loss_arr>l_th)
            if loss_idx[0].size is 0:
                break
            print('train size changes to ='+str(loss_idx[0].size))
            print("Minimum loss is="+str(min(loss_arr)))
            print("Maximum loss is="+str(max(loss_arr)))
            x_in=X_train[loss_idx]
            y_in=Y_train[loss_idx]
        history=hist11
    
    elif bmode is 'outliers_iqr':
        print('running iqr outliers generator')
        x_in=X_train
        y_in=Y_train
        hist11=[]
        stopearly1 = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
        totepoch=100
        for epp in range(1,totepoch,1):
            print("Epoch="+str(epp)+"/"+str(totepoch))
            hist1=model.fit(x_in, y_in, batch_size=batch_size, nb_epoch=1,verbose=verbose, callbacks=[stopearly1],validation_data=(X_test, Y_test) )
            hist11=hist11+hist1.history.items()+list(['tr_size',x_in.shape[0]])
            bs_=1000
            probs=model.predict_proba(X_train, batch_size=bs_,verbose=verbose)
            cross_entr=Y_train*np.log(probs)
            loss_arr=-np.sum(cross_entr,axis=1)#log loss, cross entropy error
            lt=np.percentile(loss_arr,25)
            ht=np.percentile(loss_arr,75)
            t=ht-lt
            print("IQR value 25% is="+str(lt))
            print("IQR value 75% is="+str(ht))
            print("IQR value is="+str(t))
            hist11=hist11+list(['l_iqr',lt])
            hist11=hist11+list(['h_iqr',ht])
            hist11=hist11+list(['iqr',t])

            lt=lt-t/2
            ht=ht+t/2
            print("Min threshold is="+str(lt))
            print("Max Threshold is="+str(ht))

            hist11=hist11+list(['l_thr',lt])
            hist11=hist11+list(['h_thr',ht])

            if(lt<1e-3):#0.9 probability
                lt=1e-3

            if(ht>1):#0.9 probability
                ht=1


            loss_idx=np.where(loss_arr>lt)
            
            #loss_idx=np.where(np.logical_and(loss_arr>lt,loss_arr>ht ))
            
            if loss_idx[0].size is 0:
                x_in=X_train
                y_in=Y_train
                continue
        
                
            print('train size changes to ='+str(loss_idx[0].size))
            
            lt=min(loss_arr)
            ht=max(loss_arr)
            
            print("Minimum loss is="+str(lt))
            print("Maximum loss is="+str(ht))
            hist11=hist11+list(['min',lt])
            hist11=hist11+list(['max',ht])
            
            x_in=X_train[loss_idx]
            y_in=Y_train[loss_idx]
        
        history=hist11
	
	elif bmode is 'outliers_iqr_npass':
		print('running iqr outliers generator')
		n=3
		hist11=[]
		for npass in range(1,n,1):
			x_in=X_train
			y_in=Y_train
			x_clamp=X_train
			y_clamp=Y_train			
			stopearly1 = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
			totepoch=100
			for epp in range(1,totepoch,1):
				print("Epoch="+str(epp)+"/"+str(totepoch))
				hist1=model.fit(x_in, y_in, batch_size=batch_size, nb_epoch=1,verbose=verbose, callbacks=[stopearly1],validation_data=(X_test, Y_test) )
				hist11=hist11+hist1.history.items()+list(['tr_size',x_in.shape[0]])
				bs_=1000
				if(bs_>x_clamp.shape[0]):
					bs_=x_clamp.shape[0]
				probs=model.predict_proba(x_clamp, batch_size=bs_,verbose=verbose)
				cross_entr=y_clamp*np.log(probs)
				loss_arr=-np.sum(cross_entr,axis=1)#log loss, cross entropy error
				
				nanidx=np.where(~numpy.isnan(loss_arr))#remove nan values
				print("TOTAL NAN VALUES ARE is="+str(len(nanidx)))
				loss_arr=loss_arr[nanidx]
				x_clamp=x_clamp[nanidx]
				nanidx
				lt=np.percentile(loss_arr,25)
				ht=np.percentile(loss_arr,75)
				t=ht-lt
				print("IQR value 25% is="+str(lt))
				print("IQR value 75% is="+str(ht))
				print("IQR value is="+str(t))
				hist11=hist11+list(['l_iqr',lt])
				hist11=hist11+list(['h_iqr',ht])
				hist11=hist11+list(['iqr',t])

				lt=lt-t/2
				ht=ht+t/2
				print("Min threshold is="+str(lt))
				print("Max Threshold is="+str(ht))

				hist11=hist11+list(['l_thr',lt])
				hist11=hist11+list(['h_thr',ht])

				if(lt<1e-3):#0.9 probability
					lt=1e-3
				elif(lt>0.3)#0.5 probability
					lt=0.3

				if(ht>1):#0.9 probability
					ht=1
				elif(ht<0.3)#0.5 probability
					ht=0.3


				loss_idx=np.where(loss_arr>lt)
				
				#loss_idx=np.where(np.logical_and(loss_arr>lt,loss_arr>ht ))
				
				if loss_idx[0].size is 0:
					break
			
					
				print('train size changes to ='+str(loss_idx[0].size))
				
				lt=min(loss_arr)
				ht=max(loss_arr)
				
				print("Minimum loss is="+str(lt))
				print("Maximum loss is="+str(ht))
				hist11=hist11+list(['min',lt])
				hist11=hist11+list(['max',ht])
				
				x_in=x_clamp[loss_idx]
				y_in=y_clamp[loss_idx]
			
		history=hist11

    elif bmode is 'care_outliers_iqr':
        print('running care iqr outliers generator')
        x_in=X_train
        y_in=Y_train
		
	#	x_clamp=X_train
     #   y_clamp=Y_train
        hist11=[]
        stopearly1 = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
        totepoch=100
        for epp in range(1,totepoch,1):
            print("Epoch="+str(epp)+"/"+str(totepoch))
            newgen = generator_of_new(x_in, y_if, c, batch_size)
            hist1=model.fit_generator(newgen, nb_epoch=1, verbose=verbose,callbacks=[stopearly1],validation_data=(X_test, Y_test),
                    samples_per_epoch=x_in.shape[0])
            hist11=hist11+hist1.history.items()+list(['tr_size',x_in.shape[0]])
            bs_=1000
            probs=model.predict_proba(X_train, batch_size=bs_,verbose=verbose)
            cross_entr=Y_train*np.log(probs)
            loss_arr=-np.sum(cross_entr,axis=1)#log loss, cross entropy error
            lt=np.percentile(loss_arr,25)
            ht=np.percentile(loss_arr,75)
            t=ht-lt
            print("IQR value 25% is="+str(lt))
            print("IQR value 75% is="+str(ht))
            print("IQR value is="+str(t))
            hist11=hist11+list(['l_iqr',lt])
            hist11=hist11+list(['h_iqr',ht])
            hist11=hist11+list(['iqr',t])

            lt=lt-t/2
            ht=ht+t/2
            print("Min threshold is="+str(lt))
            print("Max Threshold is="+str(ht))

            hist11=hist11+list(['l_thr',lt])
            hist11=hist11+list(['h_thr',ht])

            if(lt<1e-3):#0.9 probability
                lt=1e-3

            if(ht>1):#0.9 probability
                ht=1


            loss_idx=np.where(loss_arr>lt)
            
            #loss_idx=np.where(np.logical_and(loss_arr>lt,loss_arr>ht ))
            
            if loss_idx[0].size is 0:
                x_in=X_train
                y_in=Y_train
                continue
        
                
            print('train size changes to ='+str(loss_idx[0].size))
            
            lt=min(loss_arr)
            ht=max(loss_arr)
            
            print("Minimum loss is="+str(lt))
            print("Maximum loss is="+str(ht))
            hist11=hist11+list(['min',lt])
            hist11=hist11+list(['max',ht])
            
            x_in=X_train[loss_idx]
            y_in=Y_train[loss_idx]
        
        history=hist11
    
    elif bmode is 'outliers_randthr':
        print('running outliers-random threshold generator')
        x_in=X_train
        y_in=Y_train
        hist11=[]
        stopearly1 = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
        for epp in range(1,totepoch,1):
            print("Epoch="+str(epp)+"/"+str(totepoch))
            hist1=model.fit(x_in, y_in, batch_size=batch_size, nb_epoch=10,verbose=verbose, callbacks=[stopearly1],validation_data=(X_test, Y_test) )
            hist11=hist11+hist1.history.items()+list([x_in.shape[0]])
#            print(hist11)
            probs=model.predict_proba(X_train, batch_size=batch_size,verbose=verbose)
            cross_entr=Y_train*np.log(probs)
            loss_arr=-np.sum(cross_entr,axis=1)#log loss, cross entropy error
            l_th=abs(np.random.normal(loc=0,scale=1e-2,size=1))
            print('current threhsold is='+str(l_th))
            loss_idx=np.where(loss_arr>l_th)
            if loss_idx[0].size is 0:
                break
            print('train size changes to ='+str(loss_idx[0].size))
            print("Minimum loss is="+str(min(loss_arr)))
            print("Maximum loss is="+str(max(loss_arr)))
            x_in=X_train[loss_idx]
            y_in=Y_train[loss_idx]
        history=hist11

    elif bmode is 'care_outliers':
        print('running careful outliers generator')
        x_in=X_train
        y_in=Y_train
        y_if=y_f
        hist11=[]
        stopearly1 = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='auto')
        for epp in range(1,totepoch,1):
            print("Epoch="+str(epp)+"/"+str(totepoch))
            newgen = generator_of_new(x_in, y_if, c, batch_size)
            hist1=model.fit_generator(newgen, nb_epoch=10, verbose=verbose,callbacks=[stopearly1],validation_data=(X_test, Y_test),
                    samples_per_epoch=x_in.shape[0])
            hist11=hist11+hist1.history.items()+list([x_in.shape[0]])
#            print(hist11)
            probs=model.predict_proba(X_train, batch_size=batch_size,verbose=verbose)
            cross_entr=Y_train*np.log(probs)
            loss_arr=-np.sum(cross_entr,axis=1)#log loss, cross entropy error
            loss_idx=np.where(loss_arr>l_th)
            if loss_idx[0].size is 0:
                break
            print('train size changes to ='+str(loss_idx[0].size))
            print("Minimum loss is="+str(min(loss_arr)))
            print("Maximum loss is="+str(max(loss_arr)))
            x_in=X_train[loss_idx]
            y_in=Y_train[loss_idx]
            y_if=y_f[loss_idx]
        
        history=hist11
    
    score = 0
    return history, score


def fitModelMerge(model, X_train, Y_train, X_test, Y_test, y_f, batch_size=100, nb_epoch=100, verbose=0, c=10, bmode='normal'):
    history = History()
    stopearly = EarlyStopping(monitor='val_acc', patience=patience__, verbose=1, mode='auto')
    time.sleep(0.1)
    if bmode is 'normal':
        model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=[history,stopearly],validation_data=([X_test,X_test], Y_test))
    elif bmode is 'careful':
        newgen = generator_of_merge(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                        validation_data=([X_test, X_test], Y_test), samples_per_epoch=X_train.shape[0])
    elif bmode is 'up':
        newgen = generator_of_merge(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                            validation_data=([X_test, X_test], Y_test), samples_per_epoch=X_train.shape[0])
    elif bmode is 'carefulup':
        newgen = generator_of_merge(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                            validation_data=([X_test, X_test], Y_test), samples_per_epoch=X_train.shape[0])

    elif bmode is 'outliers':
        newgen = generator_of_merge(X_train, y_f, c, batch_size)
        model.fit_generator(newgen, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, stopearly],
                            validation_data=([X_test, X_test], Y_test), samples_per_epoch=X_train.shape[0])

    score = 0
    return history, score


def fitModelMerge3(model, X_train, Y_train, X_test, Y_test, y_f, batch_size=100, nb_epoch=100, verbose=0):
    history = History()
    stopearly = EarlyStopping(monitor='val_acc', patience=patience__, verbose=1, mode='auto')
    time.sleep(0.1)
    model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=verbose, callbacks=[history, stopearly], validation_data=([X_test, X_test, X_test], Y_test))
    score = model.evaluate([X_test, X_test, X_test], Y_test, verbose=0)
    return history, score


def runAnalysisOptimizers(netwk, x_t, y_t, x_v, y_v, y_f, b_size, epoc, v=1, loss='categorical_crossentropy',
                          optimizer='adadelta', img_h=28, img_w=28, d=1, c=10, init='uniform',bmode='normal'):
    model = netwk(loss=loss, optimizer=optimizer, img_rows=img_h, img_cols=img_w, depth=d, classes=c, init=init)

    hist, score = fitModel_(model, x_t, y_t, x_v, y_v, y_f, batch_size=b_size, nb_epoch=epoc, verbose=v, c=c,bmode=bmode)

    return hist, score


def runAnalysisOptimizersMerge(netwk, x_t, y_t, x_v, y_v, y_f, b_size, epoc, v=1, loss='categorical_crossentropy',
                               optimizer='adadelta', img_h=28, img_w=28, d=1, c=10, init='uniform',bmode='normal'):
    model = netwk(loss=loss, optimizer=optimizer, img_rows=img_h, img_cols=img_w, depth=d,
                  classes=c, init=init)
    hist, score = fitModelMerge(model, x_t, y_t, x_v, y_v, y_f, batch_size=b_size, nb_epoch=epoc, verbose=v, c=c,bmode=bmode)

    return hist, score


def runAnalysisOptimizersMerge3(netwk, x_t, y_t, x_v, y_v, y_f, b_size, epoc, v=1, loss='categorical_crossentropy',
                                optimizer='adadelta', img_h=28, img_w=28, d=1, c=10, init='uniform',bmode='normal'):
    model = netwk(loss=loss, optimizer=optimizer, img_rows=img_h, img_cols=img_w, depth=d,
                  classes=c, init=init)
    hist, score = fitModelMerge3(model, x_t, y_t, x_v, y_v, y_f, batch_size=b_size, nb_epoch=epoc, verbose=v,bmode=bmode)

    return hist, score


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
        x_t,y_t,x_v,y_v,y_f = flowers101.load_data()        
        img_h, img_w, d, nb_classes = 128, 128, 3, 102
        return img_h,img_w,d,nb_classes,x_t,y_t,x_v,y_v,y_f


    elif datastring is 'caltech_101':
        x_t,y_t,x_v,y_v,y_f = caltech101.load_data()        
        img_h, img_w, d, nb_classes = 64, 64, 3, 102
        return img_h,img_w,d,nb_classes,x_t,y_t,x_v,y_v,y_f

    elif datastring is 'stl_10':
        X_train, Y_train, X_test, Y_test,Y_flat = stl10.load_data()
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


def generate_batch(arr, orig, target, y_orig, c, size):
    idx = []
    targ = []
    tr_b = []
    for cl in range(0, c, 1):
        idx1 = np.where(target == cl)
        if (len(idx1[0]) > 0):
            idx_r = np.random.choice(idx1[0], size=1)
            tr_b.append(arr[idx_r[0]])
            idx.append(idx_r[0])
            targ.append(cl)
        else:
            idx1 = np.where(y_orig == cl)
            idx_r = np.random.choice(idx1[0], size=1)
            tr_b.append(orig[idx_r[0]])
            targ.append(cl)
    arr = np.delete(arr, idx, axis=0)
    target = np.delete(target, idx, axis=0)
    return np.array(tr_b), targ, arr, target


def generate_batch_new(arr, target, y_orig, c, size):
    idx = []
    targ = []
    tr_b = []
    for cl in range(0, c, 1):
        idx1 = np.where(target == cl)
        # print len(idx1[0])
        if (len(idx1[0]) > 0):
            idx_r = np.random.choice(idx1[0], size=1)
            # tr_b=np.append(tr_b,arr[idx_r[0]])
            tr_b.append(arr[idx_r[0]])
            idx.append(idx_r[0])
            targ.append(cl)
        else:
            idx1 = np.where(y_orig == cl)
            idx_r = np.random.choice(idx1[0], size=1)
            # tr_b=np.append(tr_b,orig[idx_r[0]])
            tr_b.append(arr[idx_r[0]])
            targ.append(cl)
    # arr=np.delete(arr,idx,axis=0)
    target[idx] = -1
    tot = np.where(target == -1)[0].size
    if (tot >= len(target)):
        target = np.empty(0)
    return np.array(tr_b), targ, arr, target


def generator_ofdb(x_t, y_t, c):
    # h,w,d,c,x_t,y_t,x_v,y_v=loadData(db)
    x_dum = x_t
    y_dum = y_t
    while 1:
        if (y_dum.size <= 0):
            x_dum = x_t
            y_dum = y_t
            # print "updating size"
        # print x_dum.shape, y_dum.shape
        x_b, y_b, x_dum, y_dum = generate_batch(x_dum, x_t, y_dum, y_t, c, c)
        # print y_b
        yield x_b, np_utils.to_categorical(y_b, c)


def generate_batch__anysize(arr, target, y_orig, c, size):
    idx = []
    targ = []
    tr_b = []
    todel = False
    while size > 0:
        target[idx] = -1
        tot = np.where(target == -1)[0].size
        if (tot >= len(target)):
            # print "COPYING"
            todel = True
            target = np.array(y_orig, copy=True)
        for cl in range(0, c, 1):
            idx1 = np.where(target == cl)
            if (len(idx1[0]) > 0):
                idx_r = np.random.choice(idx1[0], size=1)
                tr_b.append(arr[idx_r[0]])
                idx.append(idx_r[0])
                targ.append(cl)
            else:
                idx1 = np.where(y_orig == cl)
                idx_r = np.random.choice(idx1[0], size=1)
                tr_b.append(arr[idx_r[0]])
                targ.append(cl)
        size = size - c
        target[idx] = -1

    if todel is True:
        target = np.empty(0)
    return np.array(tr_b), targ, arr, target


def generator_of_new(x_t, y_t, c, size):
    y_dum = np.array(y_t, copy=True)
    while 1:
        if (y_dum.size <= 0):
            y_dum = np.array(y_t, copy=True)
        x_b, y_b, x_dum, y_dum = generate_batch__anysize(x_t, y_dum, y_t, c, size)
        yield x_b, np_utils.to_categorical(y_b, c)


def generator_of_merge(x_t, y_t, c, size):
    y_dum = np.array(y_t, copy=True)
    while 1:
        if (y_dum.size <= 0):
            y_dum = np.array(y_t, copy=True)
        x_b, y_b, x_dum, y_dum = generate_batch__anysize(x_t, y_dum, y_t, c, size)
        yield [x_b, x_b], np_utils.to_categorical(y_b, c)


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

nwDict = {
    'TWOLAYER'     :  TWOLAYER_DIR,
    #'TWOLAYER_N'   :  TWOLAYER_N_DIR,
    #'TWOLAYER_H'   :  TWOLAYER_H_DIR,

    'THREELAYER'   :  THREELAYER_DIR,
    #'THREELAYER_N' :  THREELAYER_N_DIR,
    #'THREELAYER_H' :  THREELAYER_H_DIR,

    'FIVELAYER'    :  FIVELAYER_DIR,
    #'FIVELAYER_N'  :  FIVELAYER_N_DIR,
    #'FIVELAYER_H'  :  FIVELAYER_H_DIR,
    #'MERGE_WEAK_2' :  MERGE_WEAK_2_DIR,
    #'MERGE_WEAK_3' :  MERGE_WEAK_3_DIR
}

nwFunc = {
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
    #'ADL': 'adadelta',
    'ADG': 'adagrad',
    'SGD':  sgd,
    #'ADM': 'adam',
    'ADMX': 'adamax',
    #'NDM' :'nadam',
}

initDict = {
    #'unif': 'uniform',
    #'norm': 'normal',
    #'zero': 'zero',
    #'gl_n': 'glorot_normal',
    'gl_u': 'glorot_uniform',
    'he_u': 'he_uniform',
}
#dnn_thesis_u.py
