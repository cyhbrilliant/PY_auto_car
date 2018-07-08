import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import AutoCar.Stereo.StereoDisparity as StereoDisparity
import AutoCar.Model_Segment.model_segment as model_segment
import AutoCar.FindLine.findline as findline
from AutoCar.Model_Segment import labels2 as lb

Frame0 = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
Frame1 = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
Frame0_rectified = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
Frame1_rectified = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
Disparity = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
threeD = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
Segment = np.zeros(shape=[160, 320, 3], dtype=np.uint8)
Segmentname = np.zeros(shape=[160, 320], dtype=np.uint8)
Line = np.zeros(shape=[240, 320, 3], dtype=np.uint8)
# click_x = 0
# click_y = 0

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
        print('采集双目影像')
Thread_Cam = threading.Thread(target=ReadCam, name='ReadCam')
Thread_Cam.start()

def Process():
    global Frame0
    global Frame1
    global Frame0_rectified
    global Frame1_rectified
    global Disparity
    global Segment
    global Segmentname
    global Line
    global threeD

    left_map1, left_map2, right_map1, right_map2, Q = StereoDisparity.initStereoParam()
    while True:
        Disparity, threeD, Frame0_rectified, Frame1_rectified = StereoDisparity.StereoCompute(Frame0, Frame1, left_map1, left_map2, right_map1, right_map2, Q)
        Segment, Segmentname = model_segment.Predict(Frame0_rectified[60:220, :], True)
        # Line = findline.findline(Frame0_rectified)
        # cv2.imshow("left", Frame0)
        # cv2.imshow("right", Frame1)
        # cv2.imshow("disparity", Disparity)
        # cv2.imshow("segment", Segment)
        # cv2.imshow("line", Line)
        # cv2.waitKey(1)
        print('分析双目影像')
Thread_Proc = threading.Thread(target=Process, name='Process')
Thread_Proc.start()

#di, dj is in 480,640
def get_ThreeD_Name(di, dj):
    global Segmentname
    global threeD

    di = int(di / 2)
    dj = int(dj / 2)
    print(di, dj)
    if di <= 60 or di >= 220 :
        return '请点击红线区域内', -1.0
    else:
        name = lb.labelsname[Segmentname[di-60, dj]].name
        dist = threeD[di, dj, 2]/1000.0
        print(dist)
        if not (dist < 14.0 and dist > 0):
            dist = -1.0
        return name, dist

      



welcome = tk.Tk()
welcome.title('智能车双目测量系统')
lwelpic = tk.Label(welcome)
lwelpic.pack()
welpic = cv2.imread('welcome.jpg')
# welpic = cv2.resize(welpic, dsize=(1280, 720))
r = 0.01
def show_welpic():
    global r
    global welpic
    r += 0.1
    kernel_size = 5
    # welpic = cv2.GaussianBlur(welpic, (kernel_size, kernel_size), r)
    welpicx = Image.fromarray(welpic)
    welpicx = ImageTk.PhotoImage(welpicx)
    # print(r)
    if r < 10:
        lwelpic.welpicx = welpicx
        lwelpic.configure(image=welpicx)
        lwelpic.after(1, show_welpic)
show_welpic()
welbutton = tk.Button(welcome, text='欢迎使用智能车双目测量系统！！！', bd=0, fg="red", font=("黑体", 30, "bold"), background='#ffffff', width=61, height=5, command=welcome.destroy)
welbutton.pack()
lwelpic.mainloop()




root = tk.Tk()
root.title('智能车双目测量系统')

# root_Mouse = tk.Label(root)
# root_Mouse.pack(side="left")
# root_Frame0 = tk.Label(root)
# root_Frame0.pack(side="left")
# root_Frame1 = tk.Label(root)
# root_Frame1.pack(side="left")
# root_Frame0_rectified = tk.Label(root)
# root_Frame0_rectified.pack(side="left")
# root_Frame1_rectified = tk.Label(root)
# root_Frame1_rectified.pack(side="left")
# root_Disparity = tk.Label(root)
# root_Disparity.pack()
# root_Segment = tk.Label(root)
# root_Segment.pack()
# root_Line = tk.Label(root)
# root_Line.pack()

# root_Mouse = tk.Label(root)
# root_Mouse.grid(row=0, column=0)
# root_Frame0 = tk.Label(root)
# root_Frame0.grid(row=0, column=1)
# root_Frame1 = tk.Label(root)
# root_Frame1.grid(row=0, column=2)
# root_Frame0_rectified = tk.Label(root)
# root_Frame0_rectified.grid(row=0, column=1)
# root_Frame1_rectified = tk.Label(root)
# root_Frame1_rectified.grid(row=0, column=2)
# root_Disparity = tk.Label(root)
# root_Disparity.grid(row=0, column=3)
# root_Segment = tk.Label(root)
# root_Segment.grid(row=1, column=3)
# root_Line = tk.Label(root)
# root_Line.grid(row=2, column=3)

root.geometry("1920x640+0+0")
root_Mouse = tk.Label(root, text='\n请点击红线内区域以获取物体名称与距离', compound = tk.BOTTOM, fg="red", font=("黑体", 20, "bold"))
def callback(event):
    print ("clicked at", event.x, event.y)
    name, dist = get_ThreeD_Name(event.y-30, event.x)
    dist = round(dist, 2)
    root_Result.configure(text='物体名称：'+name+'  距离：'+str(dist)+ '米', fg="red", font=("黑体", 30, "bold"))
root_Mouse.bind("<Button-1>", callback)
root_Mouse.place(x = 0, y = 0, width=640, height=480)
root_Frame0 = tk.Label(root, text='\n左摄像头', compound = tk.BOTTOM, fg="green", font=("黑体", 20, "bold"))
root_Frame0.place(x = 640, y = 0, width=320, height=240)
root_Frame1 = tk.Label(root, text='\n右摄像头', compound = tk.BOTTOM, fg="green", font=("黑体", 20, "bold"))
root_Frame1.place(x = 960, y = 0, width=320, height=240)
root_Frame0_rectified = tk.Label(root, text='\n左畸变校正', compound = tk.BOTTOM, fg="green", font=("黑体", 20, "bold"))
root_Frame0_rectified.place(x = 640, y = 240, width=320, height=240)
root_Frame1_rectified = tk.Label(root, text='\n右畸变校正', compound = tk.BOTTOM, fg="green", font=("黑体", 20, "bold"))
root_Frame1_rectified.place(x = 960, y = 240, width=320, height=240)
root_Disparity = tk.Label(root, text='\n深度三维重建', compound = tk.BOTTOM, fg="blue", font=("黑体", 20, "bold"))
root_Disparity.place(x = 1280, y = 0, width=320, height=240)
root_Segment = tk.Label(root, text='\n场景语义分割', compound = tk.BOTTOM, fg="blue", font=("黑体", 20, "bold"))
root_Segment.place(x = 1280, y = 240, width=640, height=320)
root_Line = tk.Label(root, text='\n车道线检测', compound = tk.BOTTOM, fg="blue", font=("黑体", 20, "bold"))
root_Line.place(x = 1600, y = 0, width=320, height=240)
root_Result = tk.Label(root)
root_Result.place(x = 0, y = 480, width=1280, height=160)

def Visible_Mouse():
    global Frame0_rectified
    Frame0_rectified_RGB = cv2.cvtColor(Frame0_rectified, cv2.COLOR_BGR2RGB)
    Frame0_rectified_RGB = cv2.resize(Frame0_rectified_RGB, dsize=(640, 480))
    cv2.line(Frame0_rectified_RGB, (0, 120), (640, 120), (255, 0, 0), 5)
    cv2.line(Frame0_rectified_RGB, (0, 440), (640, 440), (255, 0, 0), 5)
    cv2.line(Frame0_rectified_RGB, (2, 120), (2, 440), (255, 0, 0), 5)
    cv2.line(Frame0_rectified_RGB, (638, 120), (638, 440), (255, 0, 0), 5)
    Frame0_rectified_tk = Image.fromarray(Frame0_rectified_RGB)
    Frame0_rectified_tk = ImageTk.PhotoImage(Frame0_rectified_tk)
    root_Mouse.Frame0_rectified_tk = Frame0_rectified_tk
    root_Mouse.configure(image=Frame0_rectified_tk)
    root_Mouse.after(100, Visible_Mouse)
Visible_Mouse()

def Visible_Frame0():
    global Frame0
    Frame0_RGB = cv2.cvtColor(Frame0, cv2.COLOR_BGR2RGB)
    # Frame0_RGB = cv2.resize(Frame0_RGB, dsize=(640, 480))
    Frame0_tk = Image.fromarray(Frame0_RGB)
    Frame0_tk = ImageTk.PhotoImage(Frame0_tk)
    root_Frame0.Frame0_tk = Frame0_tk
    root_Frame0.configure(image=Frame0_tk)
    root_Frame0.after(100, Visible_Frame0)
Visible_Frame0()

def Visible_Frame1():
    global Frame1
    Frame1_RGB = cv2.cvtColor(Frame1, cv2.COLOR_BGR2RGB)
    # Frame1_RGB = cv2.resize(Frame1_RGB, dsize=(640, 480))
    Frame1_tk = Image.fromarray(Frame1_RGB)
    Frame1_tk = ImageTk.PhotoImage(Frame1_tk)
    root_Frame1.Frame1_tk = Frame1_tk
    root_Frame1.configure(image=Frame1_tk)
    root_Frame1.after(100, Visible_Frame1)
Visible_Frame1()

def Visible_Frame0_rectified():
    global Frame0_rectified
    Frame0_rectified_RGB = cv2.cvtColor(Frame0_rectified, cv2.COLOR_BGR2RGB)
    # Frame0_RGB = cv2.resize(Frame0_RGB, dsize=(640, 480))
    Frame0_rectified_tk = Image.fromarray(Frame0_rectified_RGB)
    Frame0_rectified_tk = ImageTk.PhotoImage(Frame0_rectified_tk)
    root_Frame0_rectified.Frame0_rectified_tk = Frame0_rectified_tk
    root_Frame0_rectified.configure(image=Frame0_rectified_tk)
    root_Frame0_rectified.after(100, Visible_Frame0_rectified)
Visible_Frame0_rectified()

def Visible_Frame1_rectified():
    global Frame1_rectified
    Frame1_rectified_RGB = cv2.cvtColor(Frame1_rectified, cv2.COLOR_BGR2RGB)
    # Frame0_RGB = cv2.resize(Frame0_RGB, dsize=(640, 480))
    Frame1_rectified_tk = Image.fromarray(Frame1_rectified_RGB)
    Frame1_rectified_tk = ImageTk.PhotoImage(Frame1_rectified_tk)
    root_Frame1_rectified.Frame1_rectified_tk = Frame1_rectified_tk
    root_Frame1_rectified.configure(image=Frame1_rectified_tk)
    root_Frame1_rectified.after(100, Visible_Frame1_rectified)
Visible_Frame1_rectified()

def Visible_Disparity():
    global Disparity
    Disparity_tk = Image.fromarray(Disparity)
    Disparity_tk = ImageTk.PhotoImage(Disparity_tk)
    root_Disparity.Disparity_tk = Disparity_tk
    root_Disparity.configure(image=Disparity_tk)
    root_Disparity.after(100, Visible_Disparity)
Visible_Disparity()

def Visible_Segment():
    global Segment
    Segment_RGB = cv2.cvtColor(Segment, cv2.COLOR_BGR2RGB)
    Segment_RGB = cv2.resize(Segment_RGB, dsize=(640, 320))
    Segment_tk = Image.fromarray(Segment_RGB)
    Segment_tk = ImageTk.PhotoImage(Segment_tk)
    root_Segment.Segment_tk = Segment_tk
    root_Segment.configure(image=Segment_tk)
    root_Segment.after(100, Visible_Segment)
Visible_Segment()

def Visible_Line():
    global Line
    Line_RGB = cv2.cvtColor(Line, cv2.COLOR_BGR2RGB)
    Line_tk = Image.fromarray(Line_RGB)
    Line_tk = ImageTk.PhotoImage(Line_tk)
    root_Line.Line_tk = Line_tk
    root_Line.configure(image=Line_tk)
    root_Line.after(100, Visible_Line)
Visible_Line()

root.mainloop()