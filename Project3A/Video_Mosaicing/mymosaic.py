'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

from helper import *;

def processList(img_input):



  length = int(len(img_input));

  middleElement = int(length/2);

  leftPart = img_input[:middleElement]
  leftPart.reverse()
  rightPart = img_input[middleElement + 1:]

  centerImage = img_input[middleElement]

  img_mosaic = findOutputForColorImageArray(centerImage, leftPart + rightPart)

  return img_mosaic

def mymosaic(img_input):
  # Your Code Here

  return processList(list(img_input))