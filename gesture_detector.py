import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('video feed', frame)  # show video stream#
    keyInput = cv2.waitKey(1)  # wait one millisecond for key input by user

    if keyInput == ord('q'):
        break