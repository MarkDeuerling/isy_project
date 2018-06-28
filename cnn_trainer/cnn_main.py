import numpy as np
import pandas as pd

train = pd.read_csv("../source/sign_mnist_train.csv").values
print(type(train))
# test  = pd.read_csv("../input/fashion-mnist_test.csv").values

'''
helpful links:
https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file
'''