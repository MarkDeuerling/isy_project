import cv2
import numpy as np
import keras
from keras.models import model_from_json, load_model
from keras.preprocessing import image

# load json and create model

json_file = open('cnn_trainer/good_results/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cnn_trainer/good_results/weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# print(dir(loaded_model))

# load weights into new model
# model = load_model("./cnn_trainer/model.h5")

class_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
<<<<<<< HEAD
    img = cv2.resize(frame, (28, 28))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    # prediction_array = loaded_model.predict(frame)
    classes = loaded_model.predict_classes(x, batch_size=25)
=======
    prediction_array = model.predict_classes(frame)
    prediction_array = model.predict(frame, verbose=1)
    print(prediction_array)
>>>>>>> cece0e407b5b4f9533449050d04d9f1633eda454

    als = class_map.get(classes[0])
    frame = cv2.putText(frame, als, (20, 90), font, 4, (0xff, 0xff, 0xff), 2, cv2.LINE_AA)
    cv2.imshow('ALS', frame)
    keyInput = cv2.waitKey(1)

    if keyInput == ord('q'):
        break
