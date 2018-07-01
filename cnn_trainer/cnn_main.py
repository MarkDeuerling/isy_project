import numpy as np
import pandas as pd
from cnn_trainer import cnn_model
from keras.utils import np_utils
import plot_utils

train = pd.read_csv("../source/sign_mnist_train.csv")
train_x = train.ix[:, 1:].values.astype('int32')
train_label = train.ix[:, 0].values.astype('int32')
train_y = np_utils.to_categorical(train_label)

test = pd.read_csv("../source/sign_mnist_test.csv")
test_x = test.ix[:, 1:].values.astype('int32')
test_label = test.ix[:, 0].values.astype('int32')
test_y = np_utils.to_categorical(test_label)

train_x, test_x = cnn_model.reshape_input_data(train_x, test_x)

batch_size = 25
epochs = 5

model = cnn_model.load_cnn_model()

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_x, test_y))

# save model
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#    json_file.write(model_json)

# save weight
model.save('model.h5')

score = model.evaluate(test_x, test_y, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_utils.plot_model_history(history)

'''
helpful links:
https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file
'''