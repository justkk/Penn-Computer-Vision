'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np

from constants import *

def addIntoSet(point, visitedSet, data):

  if tuple(point) not in visitedSet:
    visitedSet[tuple(point)] = 1
    data.append(point)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=MAX
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def anms_vec(cimg, max_pts):

  meanValue = np.mean(cimg.flatten()) + np.std(cimg.flatten())/2;

  validIndexes = np.where(cimg.flatten() > meanValue)


  iflat = cimg.flatten()

  identity = np.arange(iflat.shape[0])

  iflat = iflat[validIndexes]

  imap, jmap = np.meshgrid(np.arange(cimg.shape[0]),np.arange(cimg.shape[1]), indexing='ij')

  imap = imap.flatten()
  jmap = jmap.flatten()

  imap = imap[validIndexes]
  jmap = jmap[validIndexes]

  identity = identity[validIndexes]

  imap = imap.reshape((1,imap.shape[0]))
  jmap = jmap.reshape((1,jmap.shape[0]))


  iref = np.tile(imap, (imap.shape[1],1))
  jref = np.tile(jmap, (imap.shape[1],1))


  ireft = np.transpose(iref)
  jreft = np.transpose(jref)

  #print(iref)
  #print(jref)


  idistance = iref * iref + ireft * ireft - 2 * ireft * iref
  jdistance = jref * jref + jreft * jreft - 2 * jref * jreft

  #print(idistance + jdistance)
  #print(jdistance)

  distance = idistance + jdistance

  distIndexes = np.argsort(distance, axis=1)


  #print(distIndexes)

  distIndexesflatten = distIndexes.flatten()

  intensitiesOrder = iflat[distIndexesflatten]

  identityOrder = identity[distIndexesflatten]


  intensitiesOrder = intensitiesOrder.reshape((imap.shape[1], imap.shape[1]))

  identityOrder = identityOrder.reshape((imap.shape[1], imap.shape[1]))

  distanceOrder = np.sort(distance, axis = 1)




  #print(intensitiesOrder)

  intitalRow = intensitiesOrder[:,0].reshape((imap.shape[1],1))

  intitalRow = intitalRow * THRES

  intitalRowRepeat = np.tile(intitalRow, (1,imap.shape[1]))


  intensitiesOrderSub = intensitiesOrder - intitalRowRepeat

  intensitiesOrderSub[np.where(intensitiesOrderSub < 0)] = MAX


  # savior = np.zeros((imap.shape[1], imap.shape[1]))

  # savior[0,:] = MAX

  intensitiesOrderSub[:,0] = MAX

  #print(intensitiesOrder)
  #print(distanceOrder)
  #print(distanceOrder)

  #print(intensitiesOrderSub)


  # intensitySortOrder = np.argsort(intensitiesOrderSub, axis = 1)

  # print(intensitySortOrder)



  indexCount = np.arange(imap.shape[1]) * imap.shape[1]



  # rowTopper = intensitySortOrder[:,0]

  rowTopper = first_nonzero(intensitiesOrderSub[:,1:], axis = 1)  + 1

  rowTopperIndex = rowTopper + indexCount

  radius = distanceOrder.flatten()[rowTopperIndex]

  globalMaxIndex = np.where(iflat == np.max(iflat))

  radius[globalMaxIndex] = np.max(distance, axis = 1)[globalMaxIndex]

  

  radiusSortOrder = np.argsort(radius)[::-1]


  yOrder = imap.flatten()[radiusSortOrder]

  xOrder = jmap.flatten()[radiusSortOrder]

  radiusOrder = np.sort(radius)

  maxRadius = radiusOrder[max_pts - 1]

  #print(radius.reshape(cimg.shape[0], cimg.shape[1]))

  return np.array(xOrder), np.array(yOrder), maxRadius


def getRadiusMap(rows, cols):

  radius = 1; 
  radiusMap = {};
  radiusMap[0] = [np.array([0,0])]
 

  visitedSet = {}
  visitedSet[(0,0)] = 1;

  maxRadius = max(rows, cols)


  while radius < maxRadius:

    radiusMap[radius] = []

    previousRadiusLocations = radiusMap[radius-1]

    currentMap = {}

    for location in previousRadiusLocations:

      cleft = location + np.array([-1,0])
      cright = location + np.array([1,0])
      ctop = location + np.array([0,-1])
      cdown = location + np.array([0, 1])

      addIntoSet(cleft, visitedSet, radiusMap[radius])
      addIntoSet(cright, visitedSet, radiusMap[radius])
      addIntoSet(ctop, visitedSet, radiusMap[radius])
      addIntoSet(cdown, visitedSet, radiusMap[radius])

    radius +=1


  keys = radiusMap.keys()



  return radiusMap

def anms(cimg, max_pts):
  return anms_vec(cimg, max_pts)


def anmsOld(cimg, max_pts):

  rows = cimg.shape[0]
  cols = cimg.shape[1]

  radiusMap = getRadiusMap(rows, cols)

  radiusKeys = radiusMap.keys()
  maxRadius = max(radiusKeys)

  outputMatrix = [];

  


  for i in range(rows):
    for j in range(cols):

      currentPoint = np.array([i,j])

      currentValue = cimg[i,j]


      for radius in range(1,maxRadius):

        points = np.array(radiusMap[radius]) + currentPoint


        prows = points[:,0]
        pCols = points[:,1]

        boolRows = (prows >= 0)*(prows < rows)
        boolcols = (pCols >= 0)*(pCols < cols)

        indexes = np.where(boolRows*boolcols*1 == 1)

        if indexes[0].shape[0] == 0 : 
          continue

        validPoints = points[indexes[0], :]


        intensityValues = cimg[validPoints[:,0], validPoints[:,1]]

        if np.max(intensityValues) >= THRES* currentValue:
          outputMatrix.append([np.array((i,j)), radius]);
          break;


    sortedRadiusList = sorted(outputMatrix, key=lambda x: x[1], reverse=True)


    if max_pts > len(sortedRadiusList):
      max_pts = len(sortedRadiusList) - 1;

    rmax = sortedRadiusList[max_pts-1][1]

    sortedRadiusListPruned = [x[0] for x in sortedRadiusList[:max_pts]]

    sortedRadiusMatrix = np.array(sortedRadiusListPruned)

    y = sortedRadiusMatrix[:,0]
    x = sortedRadiusMatrix[:,1]


  # Your Code Here
  return x, y, rmax





img = np.zeros((5,5), dtype="float")


img[0,0] = 1

img[4,0] = 1

img[4,4] = 2

img[0,4] = 1

img[2,2] = 1

print(img)


anms_vec(img, 5)