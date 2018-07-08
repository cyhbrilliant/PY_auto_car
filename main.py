import cv2
import numpy as np
import threading
import AutoCar.Stereo.StereoDisparity as StereoDisparity
import AutoCar.Model_Segment.model_segment as model_segment
import AutoCar.FindLine.findline as findline

Frame0 = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
Frame1 = np.zeros(shape=[240, 320, 3], dtype=np.uint8)

def ReadCam():
    global Frame0
    global Frame1
    cap1 = cv2.VideoCapture(2)
    cap1.set(3, 320)
    cap1.set(4, 240)
    cap0 = cv2.VideoCapture(1)
    cap0.set(3, 320)
    cap0.set(4, 240)
    while True:
        ret, frame0 = cap0.read()
        ret, frame1 = cap1.read()
        # frame = cv2.resize(frame, dsize=(320, 240))
        Frame0 = frame0
        Frame1 = frame1

        # cv2.imshow("capture", frame)
        # cv2.waitKey(1)
        print('a')
Thread_Cam = threading.Thread(target=ReadCam, name='ReadCam')
Thread_Cam.start()

def Process():
    left_map1, left_map2, right_map1, right_map2, Q = StereoDisparity.initStereoParam()
    while True:
        Disparity, threeD = StereoDisparity.StereoCompute(Frame0, Frame1, left_map1, left_map2, right_map1, right_map2, Q)
        Segment = model_segment.Predict(Frame0[60:220, :], True)
        Line = findline.findline(Frame0)
        cv2.imshow("left", Frame0)
        cv2.imshow("right", Frame1)
        cv2.imshow("disparity", Disparity)
        cv2.imshow("segment", Segment)
        cv2.imshow("line", Line)
        cv2.waitKey(1)
        print('b')
Thread_Proc = threading.Thread(target=Process, name='Process')
Thread_Proc.start()


