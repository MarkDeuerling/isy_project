from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def reshape_input_data(x_train, x_test, row=28, cols=28):
    x_train = x_train.reshape(x_train.shape[0], row, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], row, cols, 1)
    return x_train, x_test


def load_cnn_model(classes=25):
    model = Sequential()
    # todo
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(classes))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
