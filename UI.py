import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *

import matplotlib.pyplot as plt
from FE import image_enhance, thinning, feature


class FingerPrint(QWidget):
    def __init__(self):
        super(FingerPrint, self).__init__()

        self.btn = QPushButton(self)
        self.showOrigImg = QLabel(self)
        self.labelEnhance = QLabel(self)
        self.showEnhanceImg = QLabel(self)
        self.labelThin = QLabel(self)
        self.showThinImg = QLabel(self)
        self.labelFeatureImg = QLabel(self)
        self.showFeatureImg = QLabel(self)
        self.labelResult = QLabel(self)
        self.showResult = QLabel(self)

        self.orig_img = None
        self.enhance_img = None
        self.thin_img = None
        self.feature_img = None
        self.features = None
        self.text = ''

        self.resize(1080, 750)
        self.setWindowTitle("指纹特征提取")
        self.GUInit()

    def FeatureExtractor(self, img):
        self.enhance_img = image_enhance(img)
        self.thin_img = thinning(self.enhance_img.copy())
        self.feature_img, self.features = feature(self.thin_img.copy())

    def GUInit(self):    
        # open image file

        self.btn.setText("打开图片")
        self.btn.setFixedSize(80, 30)
        self.btn.move(110, 35)
        self.btn.clicked.connect(self.openImage)

        self.showOrigImg.setFixedSize(200, 200)
        self.showOrigImg.move(50, 100)
        self.showOrigImg.setStyleSheet("QLabel{background:white;}")

        self.labelEnhance.setFixedSize(80, 30)
        self.labelEnhance.setText('图像增强')
        self.labelEnhance.move(410, 35)

        self.showEnhanceImg.setFixedSize(200, 200)
        self.showEnhanceImg.move(350, 100)
        self.showEnhanceImg.setStyleSheet("QLabel{background:white;}")

        self.labelThin.setFixedSize(80, 30)
        self.labelThin.setText('图像细化')
        self.labelThin.move(710, 35)

        self.showThinImg.setFixedSize(200, 200)
        self.showThinImg.move(650, 100)
        self.showThinImg.setStyleSheet("QLabel{background:white;}")

        self.labelFeatureImg.setFixedSize(80, 30)
        self.labelFeatureImg.setText('特征提取结果')
        self.labelFeatureImg.move(160, 335)

        self.showFeatureImg.setFixedSize(300, 300)
        self.showFeatureImg.move(50, 400)
        self.showFeatureImg.setStyleSheet("QLabel{background:white;}")

        self.labelResult.setFixedSize(80, 30)
        self.labelResult.setText('特征提取结果')
        self.labelResult.move(625, 335)

        self.showResult.setFixedSize(640, 300)
        self.showResult.move(400, 400)
        self.showResult.setStyleSheet("QLabel{background:white;}"
                                          "QLabel{color:rgb(300,300,300,120);font-size:13px;font-weight:bold;font-family:宋体;}"
                                          )
        self.showResult.setText('特征提取结果')

    def cvImgtoQtImg(self, cv_img):  # 将OpenCV图像转为PyQt图像
        qt_img_buf = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
        qt_img = QtGui.QImage(qt_img_buf.data, qt_img_buf.shape[1], qt_img_buf.shape[0], QtGui.QImage.Format_RGB32)
        return qt_img

    def openImage(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "./DB3_B", "*.tif;;*.png;;All Files(*)")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        self.FeatureExtractor(img)

        orig_img = self.cvImgtoQtImg(img)
        orig_img = QtGui.QPixmap.fromImage(orig_img).scaled(200, 200)
        self.showOrigImg.setPixmap(orig_img)

        enhance_img = np.uint8(self.enhance_img)
        enhance_img = self.cvImgtoQtImg(enhance_img)
        enhance_img = QtGui.QPixmap.fromImage(enhance_img).scaled(200, 200)
        self.showEnhanceImg.setPixmap(enhance_img)

        thin_img = np.uint8(self.thin_img)
        thin_img = self.cvImgtoQtImg(thin_img)
        thin_img = QtGui.QPixmap.fromImage(thin_img).scaled(200, 200)
        self.showThinImg.setPixmap(thin_img)

        feature_img = np.uint8(self.feature_img)
        feature_img = self.cvImgtoQtImg(feature_img)
        feature_img = QtGui.QPixmap.fromImage(feature_img).scaled(300, 300)
        self.showFeatureImg.setPixmap(feature_img)

        self.text = ''
        for feature in self.features:
            text = '坐标：(%d, %d)，类型：%s ，角度：' % (feature[0], feature[1], feature[2])
            for angle in feature[3]:
                angle = angle / 3.14 * 180
                text += '%.2f°\t' % angle
            self.text += (text + '\n')

        self.showResult.setText(self.text)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = FingerPrint()
    my.show()
    sys.exit(app.exec_())