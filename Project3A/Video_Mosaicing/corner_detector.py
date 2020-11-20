'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''

from skimage.feature import corner_harris, corner_peaks
import numpy as np
import cv2 

def corner_detector(img):
  # Your Code Here
  return corner_harris(img)

def corner_detector_cv(img):
  return cv2.cornerHarris(img,2,3,0.004)