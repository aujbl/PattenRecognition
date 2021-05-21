# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:49:30 2021

@author: 95348
"""
from math import *
#GUI模块
import tkinter as tk
from tkinter.filedialog import *

import numpy as np
import scipy
from PIL import ImageTk, Image
from cv2 import *
from scipy import ndimage
from scipy import signal

from FingerprintImageEnhancer import FingerprintImageEnhancer

# def imshow(img, name='img'):
#     cv2.imshow('img', img)
#     cv2.waitKey(0)

class FingerprintFeatureExtraction(tk.Frame):
    def __init__(self, window):
        super().__init__(window)
        self.gui = window
        self.guiInit()
        self.img_enhancer = FingerprintImageEnhancer()
        
        self.gui.mainloop()
        
    def guiInit(self):

        self.gui.title('指纹特征提取')
        self.gui.geometry('1080x720')
        
        self.btnChoose = Button(self.gui, text='选择图片', command=self.ChoosePic)
        self.btnChoose.place(x=50, y=50)
        
        self.showOrigImg = Label(self.gui)
        self.showOrigImg.place(x=50, y=100)
        
        self.btnEnhence = Button(self.gui, text='图像增强', command=self.EnhenceImg)
        self.btnEnhence.place(x=350, y=50)
        
        self.showEnhenceImg = Label(self.gui)
        self.showEnhenceImg.place(x=350, y=100)
        
        self.btnThin = Button(self.gui, text='细化')
        self.btnThin.place(x=650, y=50)
        
        self.showThinImg = Label(self.gui)
        self.showThinImg.place(x=650, y=100)
        
        self.btnFeature = Button(self.gui, text='特征提取')
        self.btnFeature.place(x=50, y=350)
        
        self.showFeatureImg = Label(self.gui)
        self.showFeatureImg.place(x=50, y=400)
        
        self.text = Text(self.gui, height=25, width=100)
        self.text.place(x=350, y=350)
        
        self.copyImg = Label(self.gui)
        
        
    def ChoosePic(self):
        img_path = askopenfilename(initialdir='./DB3_B/', title='请选择指纹图片',
                                   filetypes=[('tif', '*.tif'), ('jpg', '*.jpg'), ('png', '*.png')])
        if img_path:
            # self.img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.img = cv2.imread(img_path)
            if len(self.img.shape) > 2:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            rows, cols = self.img.shape
            self.img = cv2.resize(self.img, (200*rows//cols, 200))   
        self.showImg(self.img, self.showOrigImg)
        # print('img_path: ', img_path)
        
    
    def EnhenceImg(self):
        self.enhenceImg = self.img_enhancer.enhance(self.img, resize=False)
        self.enhenceImg = np.uint8(self.enhenceImg*255)
        print(self.enhenceImg)
        self.showImg(self.enhenceImg, self.showEnhenceImg)
    
    def showImg(self, img, label):
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.copyImg.image = img
        label.configure(image=img)

            
if __name__ == '__main__':
    window = Tk()
    featureExtra = FingerprintFeatureExtraction(window)
            
        
    
    

































