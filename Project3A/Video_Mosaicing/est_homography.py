'''
  File name: est_homography.py
  Author: Haoyuan(Steve) Zhang
  Date created: 10/15/2017
'''

'''
  File clarification:
    Estimate homography for source and target image, given correspondences in source image (x, y) and target image (X, Y) respectively
    - Input x, y: the coordinates of source correspondences
    - Input X, Y: the coordinates of target correspondences
      (X/Y/x/y , each is a numpy array of n x 1, n >= 4)

    - Output H: the homography output which is 3 x 3
      ([X,Y,1]^T ~ H * [x,y,1]^T)
'''

import numpy as np
import pdb

import cv2

def est_homography(x, y, X, Y):

  N = x.shape[0]

  sPoints = np.zeros((N,2), dtype="float32")

  dPoints = np.zeros((N,2), dtype="float32")


  sPoints[:,0] = x
  sPoints[:,1] = y

  dPoints[:,0] = X
  dPoints[:,1] = Y

  print(sPoints)
  print(dPoints)

  #h, status = cv2.findHomography(sPoints, dPoints)

  ans = cv2.getPerspectiveTransform(sPoints.astype("float32"), dPoints.astype("float32"))

  return ans


def est_homography2(x, y, X, Y):

  N = x.size
  A = np.zeros([2 * N, 9])

  i = 0
  while i < N:
    a = np.array([x[i], y[i], 1]).reshape(-1, 3)
    c = np.array([[X[i]], [Y[i]]])
    d = - c * a

    A[2 * i, 0 : 3], A[2 * i + 1, 3 : 6]= a, a
    A[2 * i : 2 * i + 2, 6 : ] = d

    i += 1
  
  # compute the solution of A
  U, s, V = np.linalg.svd(A, full_matrices=True)
  h = V[8, :]
  H = h.reshape(3, 3)

  return H