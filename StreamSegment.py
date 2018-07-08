import cv2
import threading
import numpy as np
import AutoCar.Model_Segment.model_segment as model_segment
import matplotlib.pyplot as mp

Frame = np.zeros(shape=[160, 320, 3], dtype=np.uint8)

def ReadCam():
    global Frame
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        # frame = cv2.resize(frame, dsize=(320, 240))
        Frame = frame[60:220, :]

        # cv2.imshow("capture", frame)
        # cv2.waitKey(1)
        print('a')
Thread_Cam = threading.Thread(target=ReadCam, name='ReadCam')
Thread_Cam.start()

def Process():
    while True:
        frame = model_segment.Predict(Frame, True)
        cv2.imshow("capture", frame)
        cv2.waitKey(1)
        print('b')
Thread_Proc = threading.Thread(target=Process, name='Process')
Thread_Proc.start()
