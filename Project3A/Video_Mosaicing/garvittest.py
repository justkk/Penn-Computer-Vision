import numpy as np

from corner_detector import *
from anms import *

from feat_desc import *;

from scipy import signal;

from feat_match import *;

from ransac_est_homography import *;

from helper import *;

from PIL import Image

from scipy.misc import imresize

from sift import *


SSD_THRES = 2;

Edge = 1000;


iarray = [Image.open("test_img/1R.jpg")]#, Image.open("test_img/S3.jpg") ]#, Image.open("test_img/S1.jpg"), Image.open("test_img/S6.jpg")] 
centerImage = Image.open("test_img/1M.jpg");

findOutputForColorImageArray(centerImage, iarray, True, False, 0.25)

#findOutputForColorImage(np.array(Image.open("test_img/S1.jpg")).astype("float"), np.array(Image.open("test_img/S2.jpg")).astype("float"))


# square = rgb2gray(np.array(Image.open("test_img/1M.jpg")).astype("float"))

# squareShift = rgb2gray(np.array(Image.open("test_img/1R.jpg")).astype("float"))


# #squareShift = square.copy()

# square = imresize(square, 1.0)
# squareShift = imresize(squareShift, 1.0)



# # squareShift = signal.convolve2d(square, np.array([0,0,0,0,0,0,0,0,1]).reshape((1,9)), mode="same")



# square.astype("uint8")
# squareShift.astype("uint8")

# Image.fromarray(square).show()


# # square = np.zeros([12, 12])

# # square[4:8, 4:8] = 1;


# # #square = np.arange(1,10).reshape(3,3)

# # #square = np.pad(square, ((2,2),(2,2)), mode = "constant")


# # square.astype(int)


# # squareShift = signal.convolve2d(square, np.array([1,0,0,0,0]).reshape((1,5)), mode="same")
# # #squareShift[4:8,10:12] = 1

# # print(square)
# # print(squareShift)

# #computeSift(rgb2gray(im))


# # cmx = corner_detector(square)
# # cmxS = corner_detector(squareShift)


# # print("anms computation")
# # x,y,rmax = anms(cmx, Edge)
# # xS,yS,rmaxS = anms(cmxS, Edge)
# # print("anms done")

# # newSquare = square.copy()
# # newSquareShift = squareShift.copy()

# # newSquare[y,x] = 255;
# # newSquareShift[yS, xS] = 255;


# # Image.fromarray(newSquare.astype("uint8")).show() 
# # Image.fromarray(newSquareShift.astype("uint8")).show() 

# # sift = cv2.xfeatures2d.SIFT_create()
# # kp1 = sift.detect(square,None)
# # print(kp1)



# #fd =  feat_desc(square, x, y)
# #fdS = feat_desc(squareShift, xS, yS)


# ppx, fd = computeSift(square)

# ppsx, fdS = computeSift(squareShift)

# x = ppx[:,0]
# y = ppx[:,1]

# xS = ppsx[:,0]
# yS = ppsx[:,1]

# mathingSet1 = feat_match(fdS, fd)
# mathingSet2 = feat_match(fd, fdS)

# print("matching size")

# print(mathingSet1)
# print(mathingSet2)


# map1 = {}


# smx = []
# smy = [] 
# dmx = [] 
# dmy = []

# for index in range(mathingSet1.shape[0]):
# 	if mathingSet1[index]!= -1:
# 		map1[(x[mathingSet1[index]], y[mathingSet1[index]])] = (xS[index], yS[index])


# mathes1to2 = []

# skpoints = []
# dkpoints = []

# count = 0;

# for index in range(mathingSet2.shape[0]):
# 	if mathingSet2[index]!= -1:
# 		newPoint = (xS[mathingSet2[index]], yS[mathingSet2[index]])
# 		currentPoint = (x[index], y[index])
# 		if currentPoint in map1 and map1[currentPoint] == newPoint:
# 			smx.append(x[index])
# 			smy.append(y[index])
# 			mathes1to2.append(cv2.DMatch(count, count, 1))
# 			count += 1

# 			dmx.append(xS[mathingSet2[index]])
# 			dmy.append(yS[mathingSet2[index]])

# 			skpoints.append(cv2.KeyPoint(x=x[index],y=y[index], _size=0))
# 			dkpoints.append(cv2.KeyPoint(x=xS[mathingSet2[index]], y=yS[mathingSet2[index]], _size=0))
		



# # for index in range(mathingSet.shape[0]):
# # 	if mathingSet[index]!= -1:
# # 		dmx.append(xS[index])
# # 		dmy.append(yS[index])
# # 		smx.append(x[mathingSet[index]])
# # 		smy.append(y[mathingSet[index]])

# smx = np.array(smx)
# smy = np.array(smy)
# dmx = np.array(dmx)
# dmy = np.array(dmy)


# draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = None, flags = 2)

# src_pts = np.zeros((smx.shape[0],2), dtype="float32")
# dst_pts = np.zeros((smx.shape[0],2), dtype="float32")

# src_pts[:,0] = smx
# src_pts[:,1] = smy

# dst_pts[:,0] = dmx
# dst_pts[:,1] = dmy


# img3 = cv2.drawMatches(square,skpoints,squareShift,dkpoints,mathes1to2,None,**draw_params)


# Image.fromarray(img3.astype("uint8")).show()


# H, inliners = ransac_est_homography(dmx, dmy, smx, smy, SSD_THRES)


# #H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1.0)

# #H = np.array([[1,0,-4],[0,1,0],[0,0,1]])


# print("Homography")
# print(H)


# newImage = mergeImages(square, squareShift, H)

# print(newImage.shape)

# newImage = np.clip(newImage, 0,255)

# Image.fromarray(newImage.astype("uint8")).show()

#print(np.round(newImage))






