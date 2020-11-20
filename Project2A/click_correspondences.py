'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: source image
    - Input im2: target image
    - Output im1_pts: correspondences coordiantes in the source image
    - Output im2_pts: correspondences coordiantes in the target image
'''

from cpselect import *;

import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)


def appendBoundaries(shape, array):

  array.append(np.array([0,0]))
  array.append(np.array([shape[1],shape[0]]))
  array.append(np.array([shape[1],0]))
  array.append(np.array([0,shape[0]]))


def click_correspondences(im1, im2):
  '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''

  # TODO: Your code here

  ## We need to get the rows and cols indices. 

  points1, points2 = cpselect(im1, im2);

  points1 = list(points1)
  points2 = list(points2)


  shapeCords = im1.shape

  # points1[:,0] , points1[:,1] = points1[:,1], points1[:,0]

  # points2[:,0] , points2[:,1] = points2[:,1], points2[:,0]  

  appendBoundaries(shapeCords, points1)
  appendBoundaries(shapeCords, points2)

  return np.array(points1), np.array(points2)



if __name__=="__main__":
  from PIL import Image
  import numpy as np

  im = Image.open("./test.png")

  print(np.array(im).shape)

  click_correspondences(np.array(im),np.array(im))