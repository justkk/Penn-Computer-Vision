'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import random

ITER_LIMIT = 3000

from est_homography import *

import numpy as np



def computeInliers(x1, y1, x2, y2, thresh, H):

  inliers = []

  for index in range(len(x1)):

    source3dpoint = np.array([x1[index], y1[index], 1]).reshape((3,1))
    dest3dpoint = np.array([x2[index], y2[index], 1]).reshape((3,1))

    reconstructedPoint = np.matmul(H, source3dpoint)

    reconstructedPoint = reconstructedPoint / reconstructedPoint[2,0]

    difference = (dest3dpoint - reconstructedPoint)


    SSD = np.matmul(np.transpose(difference), difference)

    SSD = SSD[0,0]

    if SSD <= thresh:
      inliers.append(1)
    else:
      inliers.append(0)

  return inliers



def computeHomography(x1, y1, x2, y2, thresh, indexes):

  homography = est_homography(x1[indexes], y1[indexes], x2[indexes], y2[indexes])
  #print(np.round(homography))
  #print(x1[indexes], y1[indexes], x2[indexes], y2[indexes])

  inliers = computeInliers(x1, y1, x2, y2, thresh, homography)

  return inliers, sum(inliers), homography



def ransac_est_homography(x1, y1, x2, y2, thresh):
  # Your Code Here


  size = x1.shape[0]

  count = 0
  bestInliers = None
  bestH = None

  print(x1, y1)


  for it in range(ITER_LIMIT):
    randomIndexes = random.sample(range(0, x1.shape[0]), 4)

    inliers, currentCount,H = computeHomography(x1, y1, x2, y2, thresh, randomIndexes)

    if currentCount > count : 
      bestInliers = inliers
      count = currentCount
      bestH = H


  indexes = np.where(np.array(bestInliers) == 1)

  # indexes = indexes[0]

  # print(bestInliers)

  # print(indexes)

  H = est_homography2(x1[indexes], y1[indexes], x2[indexes], y2[indexes])

  return bestH, bestInliers
