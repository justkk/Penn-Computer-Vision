'''
  File name: nonMaxSup.py
  Author:
  Date created:
'''

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''

import numpy as np
from scipy import ndimage

def nonMaxSup(Mag, Ori):
  # TODO: your code here

  imageShape = Mag.shape 

  xGrid, yGrid = np.meshgrid(np.arange(imageShape[0]), np.arange(imageShape[1]))

  xGridPositioned = np.transpose(xGrid)
  yGridPositioned = np.transpose(yGrid)

  yGridLinear = yGridPositioned.reshape((-1))
  xGridLinear = xGridPositioned.reshape((-1))

  cosAngle = np.cos(Ori)
  sinAngle = np.sin(Ori)

  cos180Angle = np.cos(Ori + np.pi)
  sin180Angle = np.sin(Ori + np.pi)
  
  fXCordinate = xGridPositioned - sinAngle
  fYCordinate = yGridPositioned + cosAngle

  bXCordinate = xGridPositioned - sin180Angle
  bYCordinate = yGridPositioned + cos180Angle

  fXCordinateLinear = fXCordinate.reshape((-1))
  fYCordinateLinear = fYCordinate.reshape((-1))

  bXCordinateLinear = bXCordinate.reshape((-1))
  bYCordinateLinear = bYCordinate.reshape((-1))

  fCordinates = np.array(list(map(list, zip(fXCordinateLinear, fYCordinateLinear))))
  bCordinates = np.array(list(map(list, zip(bXCordinateLinear, bYCordinateLinear))))
  nCordinates = np.array(list(map(list, zip(xGridLinear, yGridLinear))))

  MValues = Mag.reshape((-1))

  fInterpolateData =  ndimage.map_coordinates(Mag, [fXCordinateLinear, fYCordinateLinear], order = 1)
  bInterpolateData =  ndimage.map_coordinates(Mag, [bXCordinateLinear, bYCordinateLinear], order = 1)

  fbData = np.array(list(map(max, zip(fInterpolateData, bInterpolateData))))
  maxMapLinear = MValues > 1.0 * fbData
  maxMap = maxMapLinear.reshape(imageShape)

  return maxMap