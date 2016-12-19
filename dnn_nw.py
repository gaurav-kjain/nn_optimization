
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Activation, Merge
from keras.optimizers import SGD
import time

nb_filters = 64
nb_pool = 2

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
