
import cv2 as cv2
import numpy as np

def computeSift(grayImage):

	
	
	print(grayImage.shape)
	nei = np.zeros((grayImage.shape[0], grayImage.shape[1], 3 ), dtype="uint8")
	
	nei[:,:,0] = grayImage
	nei[:,:,1] = grayImage
	nei[:,:,2] = grayImage


	sift = cv2.xfeatures2d.SIFT_create(3000)
	kp1,grid_z0 = sift.detectAndCompute(grayImage,None)
	#print(kp1)
	#orb = cv.ORB_create(nfeatures = 500)
	#kp, grid_z0 = sift.detectAndCompute(nei,None)
	pts = np.array([i.pt for i in kp1])
	print(pts)
	#grid_z0m = grid_z0 - np.mean(grid_z0, axis=1).reshape((grid_z0.shape[0],1))

#	grid_z0nv = grid_z0m / np.std(grid_z0m, axis=1).reshape((grid_z0.shape[0],1))


	print("shift")
	print(pts.shape)
	#print(grid_z0.shape)

	return pts, np.transpose(grid_z0)


