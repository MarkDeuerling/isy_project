from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D


def reshape_input_data(x_train, x_test, row=28, cols=28):
    x_train = x_train.reshape(x_train.shape[0], row, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], row, cols, 1)
    return x_train, x_test


def load_cnn_model(classes=25):
    # model = Sequential()
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.20))
    # model.add(Dense(classes, activation='softmax'))



    model = Sequential()

    #input
    model.add(Conv2D(128, (3, 3), input_shape=(28, 28, 1), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
    model.add(Dense(625))  # 128 TODO: try higher auf 625
    model.add(Activation('relu'))

    #output
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

'''
http://elvera.nue.tu-berlin.de/files/1507Bochinski2017.pdf
'''