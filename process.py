import cv2
import numpy as np
from FE import image_enhance, thinning, feature
import matplotlib.pyplot as plt


img_path = 'D:/PR/PattenRecognition/DB3_B/104_2.tif'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# cv2.imshow('img', img)
# cv2.waitKey(0)

enhance_img = image_enhance(img)

thin_img = thinning(enhance_img)

feature_img, features = feature(thin_img)

print(features)
cv2.imshow('img', np.uint8(feature_img))
cv2.waitKey(0)