import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *


from FingerprintImageEnhancer import FingerprintImageEnhancer

class FingerPrint(QWidget):
    def __init__(self):
        super(FingerPrint, self).__init__()

        self.resize(900, 750)
        self.setWindowTitle("指纹特征提取")
        self.GUInit()
        
        self.image_enhencer = FingerprintImageEnhancer()
        
        
    # def EnhenceImg(self, img):
    #     img = np.uint8(img)
    #     self.enhenceImg = self.image_enhencer.enhance(img)
        
    def GUInit(self):    
        '''open image file'''
        self.btn = QPushButton(self)
        self.btn.setText("打开图片")
        self.btn.setFixedSize(80, 30)
        self.btn.move(110, 35)
        self.btn.clicked.connect(self.openimage)
        
        self.showOrigImg = QLabel(self)
        self.showOrigImg.setFixedSize(200, 200)
        self.showOrigImg.move(50, 100)
        self.showOrigImg.setStyleSheet("QLabel{background:white;}")
        
        self.labelEnhence = QLabel(self)
        self.labelEnhence.setFixedSize(80, 30)
        self.labelEnhence.setText('图像增强')
        self.labelEnhence.move(410, 35)
        
        self.showEnhenceImg = QLabel(self)
        self.showEnhenceImg.setFixedSize(200, 200)
        self.showEnhenceImg.move(350, 100)
        self.showEnhenceImg.setStyleSheet("QLabel{background:white;}")
        
        self.labelThin = QLabel(self)
        self.labelThin.setFixedSize(80, 30)
        self.labelThin.setText('图像细化')
        self.labelThin.move(710, 35)
        
        self.showThinImg = QLabel(self)
        self.showThinImg.setFixedSize(200, 200)
        self.showThinImg.move(650, 100)
        self.showThinImg.setStyleSheet("QLabel{background:white;}")
        
        self.labelFeatureImg = QLabel(self)
        self.labelFeatureImg.setFixedSize(80, 30)
        self.labelFeatureImg.setText('特征提取结果')
        self.labelFeatureImg.move(160, 335)
        
        self.showFeatureImg = QLabel(self)
        self.showFeatureImg.setFixedSize(300, 300)
        self.showFeatureImg.move(50, 400)
        self.showFeatureImg.setStyleSheet("QLabel{background:white;}")
        
        self.labelResult = QLabel(self)
        self.labelResult.setFixedSize(80, 30)
        self.labelResult.setText('特征提取结果')
        self.labelResult.move(625, 335)
        
        self.showResult = QLabel(self)
        self.showResult.setFixedSize(450, 300)
        self.showResult.move(400, 400)
        self.showResult.setStyleSheet("QLabel{background:white;}"
                                          "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                          )
        self.showResult.setText('特征提取结果')

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.tif;;*.png;;All Files(*)")
        img = cv2.imread(imgName, 0)
        self.img = QtGui.QPixmap(imgName).scaled(200, 200)
        # self.label.setPixmap(jpg)
        self.showOrigImg.setPixmap(self.img)
        
        # self.EnhenceImg(img)
        self.enhenceImg = QtGui.QPicture(img)
        self.showEnhenceImg.setPicture(self.enhenceImg)
        # self.showEnhenceImg.setPixmap(self.enhenceImg)
        self.showThinImg.setPixmap(self.img)
        
        self.img = QtGui.QPixmap(imgName).scaled(300, 300)
        self.showFeatureImg.setPixmap(self.img)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = FingerPrint()
    my.show()
    sys.exit(app.exec_())