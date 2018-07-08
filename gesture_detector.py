import cv2
import numpy as np
import keras
from keras.models import model_from_json, load_model

# load json and create model
'''
json_file = open('./cnn_trainer/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
'''

# load weights into new model
model = load_model("./cnn_trainer/model.h5")


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prediction_array = model.predict_classes(frame)
    prediction_array = model.predict(frame, verbose=1)
    print(prediction_array)

    cv2.imshow('video feed', frame)  # show video stream#
    keyInput = cv2.waitKey(1)  # wait one millisecond for key input by user

    if keyInput == ord('q'):
        break
