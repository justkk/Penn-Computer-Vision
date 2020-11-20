'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: source image
    - Input im2: target image
    - Input im1_pts: correspondences coordiantes in the source image
    - Input im2_pts: correspondences coordiantes in the target image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''
import numpy as np
from scipy.spatial import Delaunay
from helpers import *

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # TODO: Your code here
  # Tips: use Delaunay() function to get Delaunay triangulation;
  # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.

  sourceImage = im1

  destImage = im2

  gamma = 1 - warp_frac

  dissolve = 1 - dissolve_frac

  sourcePoints = im1_pts

  destinationPoints = im2_pts

  avgPoints = (sourcePoints + destinationPoints)/2

  triStructure = TriangulationStructure(avgPoints)


  morphed_im = np.zeros((len(gamma), sourceImage.shape[0], sourceImage.shape[1], 3))


  for index in range(len(gamma)):
    print(index)
    image = np.zeros((sourceImage.shape[0], sourceImage.shape[1], 3));
    for chal in range(3):
      ot = processChannel(sourceImage[:,:,chal], destImage[:,:,chal], sourcePoints, destinationPoints, avgPoints, triStructure, gamma[index], dissolve[index])
      image[:,:,chal] = ot

    morphed_im[index,:,:,:] = image

	
  return morphed_im
