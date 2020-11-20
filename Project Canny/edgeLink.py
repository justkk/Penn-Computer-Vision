'''
  File name: edgeLink.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use hysteresis to link edges based on high and low magnitude thresholds
    - Input M: H x W logical map after non-max suppression
    - Input Mag: H x W matrix represents the magnitude of gradient
    - Input Ori: H x W matrix represents the orientation of gradient
    - Output E: H x W binary matrix represents the final canny edge detection map
'''
from constants import *
from scipy.spatial import distance
import queue
from PIL import Image
import itertools
import scipy.stats


def nearestCell( sourceCell, nextCell, imageShape):
  posibileOrientation = [[-1,-1],[-1,0],[1,1] ,[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,-1]]
  valiedCombinations = [sourceCell + np.array(combination) for combination in posibileOrientation]
  filterCombinations = [ (valiedCombination, distance.euclidean(valiedCombination, nextCell)) for valiedCombination in valiedCombinations if (valiedCombination[0] >=0 and valiedCombination[0]<imageShape[0] and valiedCombination[1]>=0 and valiedCombination[1]<imageShape[1])]
  filterCombinations.sort(key=lambda tup: tup[1])
  return filterCombinations[0][0]

def getDirectionIndex(angle):
  return int(np.round(angle/(np.pi/8)))+ 10
  allowedAngles = np.arange(-2*np.pi, 2*np.pi+0.01, np.pi/8)
  distance = [(i,np.sum(abs(allowedAngles[i]-angle))) for i in range(len(allowedAngles))]
  distance.sort(key = lambda tup: tup[1])
  return distance[0][0]

def getTopandBottomPixel(strongPixel, orientation, imageShape):

  rowIndex = strongPixel[0]
  colIndex = strongPixel[1]

  topOrientation =  orientation + np.pi/2
  bottomOrientation = orientation - np.pi/2

  topRowValue = rowIndex - np.sin(topOrientation)
  topColValue = colIndex + np.cos(topOrientation)

  bottomRowValue = rowIndex  - np.sin(bottomOrientation)
  bottomColValue = colIndex + np.cos(bottomOrientation)

  topPixel =  nearestCell(strongPixel, (topRowValue, topColValue), imageShape)
  bottomPixel = nearestCell(strongPixel, (bottomRowValue, bottomColValue), imageShape)

  return tuple(topPixel), tuple(bottomPixel)


def edgeLinkVectorized(M, Mag, orientationMap, HIGHER_THRESHOLD, LOWER_THRESHOLD):

  maxPixelMap = (M * 1) * Mag

  strongPixelMask = maxPixelMap >= (HIGHER_THRESHOLD*np.amax(Mag)) 
  weakPixelMask  = (strongPixelMask ^ 1) * maxPixelMap > (LOWER_THRESHOLD*np.amax(Mag))

  strongPixelMap = strongPixelMask*1
  weakPixelMap = weakPixelMask * 2 

  rowIndexs = np.arange(strongPixelMap.shape[0])
  colIndexs = np.arange(strongPixelMap.shape[1])

  rowMesh, colMesh = np.meshgrid(rowIndexs, colIndexs)

  outputMask = Mag <0 

  rowMesh = np.transpose(rowMesh)
  colMesh = np.transpose(colMesh)

  topOrientation = orientationMap + np.pi/2
  bottomOrientation = orientationMap - np.pi/2

  topRowValue = np.round(rowMesh - np.sin(topOrientation)).astype(int)
  topColValue = np.round(colMesh + np.cos(topOrientation)).astype(int)

  bottomRowValue = np.round(rowMesh - np.sin(bottomOrientation)).astype(int)
  bottomColValue = np.round(colMesh + np.cos(bottomOrientation)).astype(int)

  topRowValue = (topRowValue < 0) * rowMesh +  topRowValue * (topRowValue >= 0)
  topColValue = (topColValue < 0) * colMesh + topColValue * (topColValue >=0)

  topRowValue = (topRowValue >= strongPixelMap.shape[0]) * rowMesh +  topRowValue * (topRowValue < strongPixelMap.shape[0])
  topColValue = (topColValue >= strongPixelMap.shape[1]) * colMesh + topColValue * (topColValue < strongPixelMap.shape[1])

  bottomRowValue = (bottomRowValue < 0) * rowMesh +  bottomRowValue * (bottomRowValue >= 0)
  bottomColValue = (bottomColValue < 0) * colMesh +  bottomColValue * (bottomColValue >=0)

  bottomRowValue = (bottomRowValue >= strongPixelMap.shape[0]) * rowMesh +  bottomRowValue * (bottomRowValue < strongPixelMap.shape[0])
  bottomColValue = (bottomColValue >= strongPixelMap.shape[1]) * colMesh +  bottomColValue * (bottomColValue < strongPixelMap.shape[1])

  orientationLabel = np.round((orientationMap + np.pi) / (np.pi/8)).astype(float) + 1


  while(np.sum(strongPixelMask*1) > 0):
    outputMask = outputMask | strongPixelMask
    strongOrientationLabel =  (strongPixelMask*1) * orientationLabel
    topOrientationLabel = strongOrientationLabel[topRowValue.reshape((-1)), topColValue.reshape((-1))]
    bottomOrientationLabel = strongOrientationLabel[bottomRowValue.reshape((-1)), bottomColValue.reshape((-1))]
    topOrientationLabel = topOrientationLabel.reshape(weakPixelMask.shape)
    bottomOrientationLabel = bottomOrientationLabel.reshape(weakPixelMask.shape)
    strongPixelMask =  weakPixelMask & (((weakPixelMask * orientationLabel) == topOrientationLabel) | ((weakPixelMask * orientationLabel) == bottomOrientationLabel))
    weakPixelMask = ((weakPixelMask*1 - strongPixelMask*1) > 0) ;

  return outputMask.astype("uint8")


def edgeLink(M, Mag, Ori):

  edgeMask = edgeLinkVectorized(M, Mag, Ori, HIGHER_THRESHOLD, LOWER_THRESHOLD);

  edgeMap = edgeMask * 1;

  return edgeMap.astype("uint8")





    






    












