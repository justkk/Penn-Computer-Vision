import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from PIL import Image

# import functions
from findDerivatives import findDerivatives
from nonMaxSup import nonMaxSup
from edgeLink import *
from helper import *
from Test_script import Test_script
import utils, helpers
from constants import *
from helper import *

import sys


import os

def getEdgeMapFromParams(I_rgb, threshold, lineMap, colorMap):

	edgeMap = None

	if colorMap == "C":

		r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
		edgeMap = np.zeros(I_rgb.shape, dtype=float)

		rEdgeMap, rMag = getEdgeMap(r, lineMap, threshold)
		gEdgeMap, gMag= getEdgeMap(g, lineMap, threshold)
		bEdgeMap, bMag = getEdgeMap(b, lineMap, threshold)

		totalMag = rMag + gMag + bMag

		rEdgeMap = rEdgeMap * rMag * 1.0 / totalMag
		gEdgeMap = gEdgeMap * gMag * 1.0 / totalMag
		bEdgeMap = bEdgeMap * bMag * 1.0 / totalMag

		im_gray = utils.rgb2gray(I_rgb)
		normalEdgeMap, mag = getEdgeMap(im_gray, lineMap, threshold)

		edgeMap[:,:,0] = (rEdgeMap + 0.0* normalEdgeMap) 
		edgeMap[:,:,1] = (gEdgeMap + 0.0* normalEdgeMap)
		edgeMap[:,:,2] = (bEdgeMap + 0.0* normalEdgeMap)


	else:
		im_gray = utils.rgb2gray(I_rgb)
		edgeMapbinary, mag = getEdgeMap(im_gray, lineMap, threshold)
		edgeMap = np.zeros(I_rgb.shape, dtype=float)
		edgeMap[:,:,0] = edgeMapbinary
		edgeMap[:,:,1] = edgeMapbinary
		edgeMap[:,:,2] = edgeMapbinary



	edgeMap = edgeMap * 255;
	edgeMap = edgeMap.astype("uint8")

	# Image.fromarray(edgeMap.astype("uint8")).show()

	return edgeMap


def getEdgeMap(im_gray, lineMap, threshold):

	Mag, Magx, Magy, Ori = findDerivatives(im_gray)
	
	M = nonMaxSup(Mag, Ori)
	
	E = None
	
	if threshold == "L":
		E = edgeLinkLocalThreshold(M, Mag, Ori)

	elif threshold == "G":
		E = edgeLinkGlobalThreshold(M, Mag, Ori)

	elif threshold == "S":
		E = edgeLink(M, Mag, Ori)

	output = E
	
	if lineMap == "L":
		output = getLineGroup(E, Ori, LINE_INFORMATION_LIMIT)

	return output, Mag

if __name__ == "__main__":
  # the folder name that stores all images
  # please make sure that this file has the same directory as image folder
  folder = '/home/cis581/Desktop/Assignments/HW1/Project/canny_dataset'

  lineMap = sys.argv[2]
  threshold = sys.argv[1]
  colorMap = sys.argv[3]


  if threshold not in ["L","S","G"] or lineMap not in ["L","N"] or colorMap not in ["C","N"] :
  	raise(ValueError("Wrong Params"))

  #folder = '/home/cis581/Desktop/test'

  # read images one by one
  for filename in os.listdir(folder):
    # read in image and convert color space for better visualization
    im_path = os.path.join(folder, filename)
    I = np.array(Image.open(im_path).convert('RGB'))
    
    ## TO DO: Complete 'cannyEdge' function
    E = getEdgeMapFromParams(I, threshold, lineMap, colorMap)

    try:
    	os.mkdir("challenge_results/")
    except:
    	pass
    try:
    	os.mkdir("_".join(["challenge_results/challenge_results",threshold, lineMap, colorMap]))
    except:
    	pass
    im = Image.fromarray(E)
    im.save( "_".join(["challenge_results/challenge_results",threshold, lineMap, colorMap]) + "/" + "_".join([filename.split(".")[0], threshold, lineMap, colorMap ]) +".png")







