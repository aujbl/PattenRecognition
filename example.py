# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""
import numpy as np
import sys
from FingerprintImageEnhancer import FingerprintImageEnhancer
import cv2

# if __name__ == '__main__':

#     image_enhancer = FingerprintImageEnhancer()         # Create object called image_enhancer
#     if(len(sys.argv)<2):                                # load input image
#         print('loading sample image');
#         img_name = '2.jpg'
#         img = cv2.imread('../images/' + img_name)
#     elif(len(sys.argv) >= 2):
#         img_name = sys.argv[1];
#         img = cv2.imread('../images/' + img_name)

#     if(len(img.shape)>2):                               # convert image into gray if necessary
#          img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     out = image_enhancer.enhance(img)     # run image enhancer
#     image_enhancer.save_enhanced_image('../enhanced/' + img_name)   # save output
if __name__ == '__main__':
    img_path = 'D:/PR/ex4/FingerprintFeatureExtractionAndDescription-master/DB3_B/102_7.tif'
    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.imread(img_path)
    if(len(img.shape)>2):                               # convert image into gray if necessary
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    image_enhancer = FingerprintImageEnhancer()
    out = image_enhancer.enhance(img)

    out
    
    





