# -*- coding:utf-8 -*-
import tkinter
import tkinter.filedialog
import os
from PIL import Image
from time import sleep
from tkinter import *

from PIL import Image,ImageTk
from tkinter.filedialog import askopenfilename
import json
import codecs

# 创建tkinter主窗口
root = tkinter.Tk()
root.title('图片处理')

# 指定主窗口位置与大小
root.geometry('900x600')   # width x height + widthoffset + heightoffset

# 不允许改变窗口大小
root.resizable(TRUE, TRUE)
root.focusmodel()

# 定义坐标显示位置
xy_text = StringVar()

#打开图片文件并显示
def choosepic():
    path_ = askopenfilename()
    path.set(path_)
    img_open = Image.open(file_entry.get())
    img = ImageTk.PhotoImage(img_open)
    image_label.config(image=img)
    image_label.image = img  # keep a reference
def input():
    fname = tkinter.filedialog.asksaveasfilename(title=u'保存文件', filetypes=[("PNG", ".png"),("GPF",".gpf"),("JPG",".jpg"),("python",".py")])
    img=Image.open(file_entry.get())
    img.save(str(fname) + ".jpg")
def test():
    path_ = askopenfilename()

path = StringVar()
tkinter.Button(root, text = '打开文件' ,command=choosepic).place(x = 100, y = 530,w = 150, h = 40)
tkinter.Button(root, text = '输入图片' ,command=input).place(x = 350, y = 530,w = 150, h = 40)
tkinter.Button(root, text = '开始检测' ,command=test).place(x = 600, y = 530,w = 150, h = 40)
tkinter.Button(root, text = '显示结果' ,command=test).place(x = 750, y = 530,w = 150, h = 40)
file_entry = Entry(root, state='readonly', text=path)


file_entry.pack()
image_label = Label(root, bg = 'gray')
image_label.place(x=0, y=0,width = 900, height = 500)


# 启动消息主循环
root.update()
root.mainloop()