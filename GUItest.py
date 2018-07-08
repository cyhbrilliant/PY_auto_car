import tkinter as tk
import cv2
from PIL import Image, ImageTk
# root = tk.Tk()


#
# text = tk.Label(root, text='111', background='#000000')
# text.pack()
#
# li     = ['C','python','php','html','SQL','java']
# movie  = ['CSS','jQuery','Bootstrap']
# listb  = tk.Listbox(root)          #  创建两个列表组件
# listb2 = tk.Listbox(root)
# for item in li:                 # 第一个小部件插入数据
#     listb.insert(0,item)
#
# for item in movie:              # 第二个小部件插入数据
#     listb2.insert(0,item)
#
# listb.pack()                    # 将小部件放置到主窗口中
# listb2.pack()

# logo = tk.PhotoImage(file="./line.png")
# w1 = tk.Label(root, image=logo).pack(side="right")
# explanation = """At present, only GIF and PPM/PGM
# formats are supported, but an interface
# exists to allow additional image file
# formats to be added easily."""
# w2 = tk.Label(root,
#            justify=tk.LEFT,
#            padx = 10,
#            text=explanation).pack(side="left")

# logo = tk.PhotoImage(file="./line.png")
# explanation = """At present, only GIF and PPM/PGM
# formats are supported, but an interface
# exists to allow additional image file
# formats to be added easily."""
# # w = tk.Label(root,
# #           compound = tk.CENTER,
# #           text=explanation,
# #           image=logo).pack(side="right")
# w = tk.Label(root,
#           justify=tk.LEFT,
#           compound = tk.TOP,
#           padx = 10,
#           text=explanation,
#           image=logo).pack(side="right")



# counter = 0
# def counter_label(label):
#     def count():
#         global counter
#         counter += 1
#         label.config(text=str(counter))
#         label.after(1000, count)
#
#     count()
#
# root.title("Counting Seconds")
# label = tk.Label(root, fg="green")
# label.pack(side='right')
# counter_label(label)
# button = tk.Button(root, text='Stop', width=25, command=root.destroy)
# button.pack()

# import cv2
# import PIL.Image as Image
# import PIL.ImageTk as ImageTk
#
# root.title("Counting Seconds")
# label = tk.Label(root)
# label.pack()
#
# img = cv2.imread('line2.jpg')
# kernel_size = 5
# imgb = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
# imgb = Image.fromarray(imgb)
# imgb = ImageTk.PhotoImage(imgb)
# label.config(image=imgb)
# r = 0.01
# def count():
#     global r
#     global img
#
#     r += 0.1
#     imgb = cv2.GaussianBlur(img, (kernel_size, kernel_size), r)
#     imgb = Image.fromarray(imgb)
#     imgb = ImageTk.PhotoImage(imgb)
#     label.config(image=imgb)
#     label.after(1000, count)
#
#
# button = tk.Button(root, text='Stop', width=25, command=root.destroy)
# button.pack()
#
# root.mainloop()                 # 进入消息循环

# import tkinter as tk

#
# width, height = 800, 600
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#
# root.bind('<Escape>', lambda e: root.quit())
# lmain = tk.Label(root)
# lmain.pack()
#
# def show_frame():
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(10, show_frame)
#
# show_frame()
# root.mainloop()


# root.bind('<Escape>', lambda e: root.quit())
# lmain = tk.Label(root)
# lmain.pack()
# img = cv2.imread('line2.jpg')
# r = 0.01
# def show_frame():
#     global r
#     global img
#     r += 0.3
#     kernel_size = 5
#     img = cv2.GaussianBlur(img, (kernel_size, kernel_size), r)
#     imgbx = Image.fromarray(img)
#     imgbx = ImageTk.PhotoImage(imgbx)
#     print(r)
#     if r < 10:
#         lmain.imgbx = imgbx
#         lmain.configure(image=imgbx)
#         lmain.after(10, show_frame)
#
# show_frame()



# def a():
#     imgbx = Image.fromarray(img)
#     imgbx = ImageTk.PhotoImage(imgbx)
#     lmain.imgbx = imgbx
#     lmain.configure(image=imgbx)
# button = tk.Button(root, text='Stop', width=25, command=a)
# button.pack()


# root.mainloop()



# def hello(event):
#     print("Single Click, Button-l")
# def quit(event):
#     print("Double Click, so let's stop")
#     import sys; sys.exit()
#
# widget = tk.Button(None, text='Mouse Clicks')
# widget.pack()
# widget.bind('&lt;Button-1&gt;', hello)
# widget.bind('&lt;Double-1&gt;', quit)
#
#
# def motion(event):
#   print("Mouse position: (%s %s)" % (event.x, event.y))
#   return
#
# master = tk.Tk()
# whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"
# msg = tk.Message(master, text = whatever_you_do)
# msg.config(bg='lightgreen', font=('times', 24, 'italic'))
# msg.bind('&lt;Motion&gt;',motion)
# msg.pack()
# master.mainloop()
# widget.mainloop()

# root = tk.Tk()
#
# def key(event):
#     print ("pressed", repr(event.char))
#
# def callback(event):
#     print ("clicked at", event.x, event.y)
#
# frame = tk.Frame(root, width=100, height=100)
# frame.bind("<Key>", key)
# frame.bind("<Button-1>", callback)
# frame.pack()
#
# root.mainloop()