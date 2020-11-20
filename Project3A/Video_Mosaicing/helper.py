




import numpy as np

import math

from scipy.interpolate import interp2d

from scipy.interpolate import griddata


from PIL import Image

import cv2


from constants import *

from scipy import signal


import numpy as np

from corner_detector import *
from anms import *

from feat_desc import *;

from scipy import signal;

from feat_match import *;

from ransac_est_homography import *;

from PIL import Image

from scipy.misc import imresize

from sift import *

import numpy as np

from corner_detector import *

from anms import *

from feat_desc import *;

from scipy import signal;

from feat_match import *;

from ransac_est_homography import *;

from PIL import Image

from scipy.misc import imresize

from sift import *

PROWS  = 20
PCOLS  = 20

DIVIDE_EPSILON = 0.0001

from constants import *

from scipy import signal

from helper import *

from scipy import ndimage

import matplotlib.pyplot as plt

import cv2


def implOneGenerateD(img, i, j):

  rows = img.shape[0]
  cols = img.shape[1]


  rr, cc = np.meshgrid(np.arange(0,rows), np.arange(0,cols), indexing='ij')

  rr = rr.flatten()
  cc = cc.flatten()

  allPoints = np.zeros((rr.shape[0], 2));

  allPoints[:,0] = rr;
  allPoints[:,1] = cc;


  direction = [-3,-2,-1,0,1,2,3,4]

  comb = itertools.product(direction, direction)

  points = [];
  for k in comb:
    points.append(np.array(k) + np.array([i,j]))

  pointsArray = np.array(points)

  pRow = pointsArray[:,0]
  pCols = pointsArray[:,1]

  print("computing")

  grid_z0 = griddata(allPoints, img.flatten(), (pRow, pCols), fill_value='0')

  print("done")

  grid_z0m = grid_z0 - np.mean(grid_z0)

  grid_z0nv = grid_z0m / np.std(grid_z0m)

  return grid_z0nv


def implOneGenerateDALL(img, iArray, jArray, orientation):

  rows = img.shape[0]
  cols = img.shape[1]


  givenOrientation = orientation[iArray, jArray]


  rr, cc = np.meshgrid(np.arange(0,rows), np.arange(0,cols), indexing='ij')

  #print(rr,cc)

  rr = rr.flatten()
  cc = cc.flatten()

  allPoints = np.zeros((rr.shape[0], 2));

  allPoints[:,0] = rr;
  allPoints[:,1] = cc;


  direction = np.arange(-4,4)

  #direction = [0,1]

  comb = itertools.product(direction, direction)

  points = [];
  for k in comb:
    points.append(np.array(k)*5 + np.array([0,0]))

  pointsArray = np.array(points)

  pointsPatchi, pointsPatchj  = np.meshgrid(np.arange(-PROWS, PROWS,5), np.arange(-PCOLS, PCOLS,5), indexing='ij')

  pointsPatchi = pointsPatchi.flatten()

  pointsPatchj = pointsPatchj.flatten()

  print(pointsPatchi, pointsPatchj)

  pointsArray = np.zeros((pointsPatchi.shape[0], 2), dtype="float")

  pointsArray[:,0] = pointsPatchi
  pointsArray[:,1] = pointsPatchj

  pointsRadius = np.sqrt(np.sum(pointsArray * pointsArray, axis = 1))

  pointsAngle = np.arctan2(pointsArray[:,0], (pointsArray[:,1] ))


  size = pointsArray.shape[0]

  iArrayRepeat = np.repeat(iArray, size)
  jArrayRepeat = np.repeat(jArray, size)

  givenOrientationRepeat = np.repeat(givenOrientation, size)


  pointsArrayiRepeat = np.tile(pointsArray[:,0], iArray.shape[0])
  pointsArrayjRepeat = np.tile(pointsArray[:,1], iArray.shape[0])

  pointsRadiusRepeat = np.tile(pointsRadius, iArray.shape[0])
  pointsAngleRepeat = np.tile(pointsAngle, iArray.shape[0]) 

  totalAngle = pointsAngleRepeat - givenOrientationRepeat


  pRow = iArrayRepeat + pointsRadiusRepeat * np.sin(totalAngle)

  pCols = jArrayRepeat + pointsRadiusRepeat * np.cos(totalAngle)

  print((pRow),(pCols))

  #pRow = iArrayRepeat + pointsArrayiRepeat
  #pCols = jArrayRepeat + pointsArrayjRepeat


  print("computing")

  combinedPoints = np.zeros((pRow.shape[0],2), dtype="float")

  combinedPoints[:,0] = pRow

  combinedPoints[:,1] = pCols

  #print(pRow, pCols)

  grid_z0 = ndimage.map_coordinates(img, [pRow.tolist(), pCols.tolist()], order=1, mode="reflect")

  #grid_z0 = griddata(allPoints, img.flatten(), (pRow, pCols), fill_value='0')

  print("done")


  grid_z0 = grid_z0.reshape((iArray.shape[0], size))


  print("Descriptor Shape")
  print(grid_z0.shape)


  grid_z0m = grid_z0 - np.mean(grid_z0, axis=1).reshape((grid_z0.shape[0],1))

  grid_z0nv = grid_z0m / np.std(grid_z0m, axis=1).reshape((grid_z0.shape[0],1))

  return grid_z0nv


def implOne(img, x,y):

  rows = y;
  cols = x; 

  patches = []

  # for index in range(len(rows)):
  #     patch = implOneGenerateD(img, rows[index], cols[index]);
  #     patches.append(patch)

  # desc = np.array(patches)

  a,b,c, ori = findDerivatives(img)

  blurredGradientKernel = GaussianPDF_2D(GAUSSIAN_MEAN, GAUSSIAN_VAR, GAUSSIAN_SIZE_ROWS, GAUSSIAN_SIZE_COLS)

  ori =  signal.convolve2d(ori, blurredGradientKernel, 'same')

  img = signal.convolve2d(img, blurredGradientKernel, 'same')

  #ori = ori * 0 + np.pi/2;

  desc = implOneGenerateDALL(img, rows, cols, ori - np.pi/2)

  return np.transpose(desc)



def mergeImages(im1, im2, H, wrapFlag):

	xmin = 0 
	xmax = im2.shape[1] - 1

	ymin = 0
	ymax = im2.shape[0] - 1


	boundaryPoints = [np.array([xmin, ymin,1]).reshape(3,1).astype("float32"), np.array([xmin, ymax,1]).reshape(3,1).astype("float32"), \
	np.array([xmax, ymin,1]).reshape(3,1).astype("float32"), np.array([xmax, ymax,1]).reshape(3,1).astype("float32")]

	# oldPoints = [np.array([xmin, ymin]), np.array([xmin, ymax]),np.array([xmax, ymin]), np.array([xmax, ymax])]

	# print(cv2.perspectiveTransform(np.array([np.array(oldPoints).astype("float32")]), H))


	newPoints = [];

	ytop = 0
	ybottom = im1.shape[0] - 1

	xleft = 0
	xright = im1.shape[1] - 1 

	for point in boundaryPoints:
		print(point)
		newPoint = np.matmul(H, point)
		print(newPoint)
		newPoint = newPoint / newPoint[2,0]
		newPoints.append(newPoint)
		print(newPoint)

		ytop = min(ytop, newPoint[1,0])

		ybottom = max(ybottom, newPoint[1,0])

		xleft = min(xleft, newPoint[0,0])

		xright = max(xright, newPoint[0,0])


	print(ytop)
	print(ybottom)
	print(xleft)
	print(xright)

	newPoints = np.array(newPoints)

	

	print("YTOP, XLEFT")
	print(ytop, xleft)
	print(ybottom - im1.shape[0]+1, xright - im1.shape[1]+1)


	ytopabs = ytop; 

	xleftabs = xleft;


	ytop = -1 * int(math.floor(ytop))
	xleft = -1 * int(math.floor(xleft))

	ybottom = math.ceil(ybottom)
	xright = math.ceil(xright)


	newHomography = H.copy()

	print("old Homography")

	print(H)

	print("newHomography")

	defaultTranslationHomography = np.zeros((3,3), dtype="float")

	defaultTranslationHomography[0,0] = 1.0
	defaultTranslationHomography[1,1] = 1.0
	defaultTranslationHomography[2,2] = 1.0
	defaultTranslationHomography[1,2] = - ytopabs
	defaultTranslationHomography[0,2] = - xleftabs

	print(defaultTranslationHomography)


	newHomography = np.matmul(defaultTranslationHomography, H)

	print(newHomography)

	newPoints2 = []

	for point in boundaryPoints:
		newPoint = np.matmul(newHomography, point)
		newPoint = newPoint / newPoint[2,0]
		newPoints2.append(newPoint)

	newPoints2 = np.array(newPoints2)

	print("newPoints2")

	print(np.round(newPoints2))


	boundingBoxymin = math.floor(np.min(newPoints2[:,1]));
	boundingBoxxmin = math.floor(np.min(newPoints2[:,0]));

	boundingBoxymax = math.ceil(np.max(newPoints2[:,1]))
	boundingBoxxmax = math.ceil(np.max(newPoints2[:,0]))

	totalY  = math.ceil(boundingBoxymax);
	totalX = math.ceil(boundingBoxxmax) ;

	print("boundings")

	print(np.round(boundingBoxymin), np.round(boundingBoxxmin), np.round(boundingBoxymax), np.round(boundingBoxxmax))


	startingOffsetx =  boundingBoxxmin 
	startingOffsety =  boundingBoxymin

	print("startingOffset")

	print(startingOffsetx, startingOffsety)


	sourceMask = (im1 > 0) * 255;


	newImage = np.pad(im1, ((ytop, ybottom - im1.shape[0] + 1), (xleft, xright  - im1.shape[1] + 1)), mode = "constant")

	sourceMask = np.pad(sourceMask, ((ytop, ybottom - im1.shape[0] + 1), (xleft, xright  - im1.shape[1] + 1)), mode = "constant")


	newImage = newImage.astype("float")

	print(newImage.shape)

	temp = None
	mask = None

	if wrapFlag:

		newHomographyInverse = np.linalg.inv(newHomography)

		print("Inverse newHomography")

		print(newHomographyInverse)

		#newHomographyInverse = newHomography


		xx, yy = np.meshgrid(np.arange(0,im2.shape[1]), np.arange(0,im2.shape[0]))
		xx= xx.flatten()
		yy = yy.flatten()

		z = im2.flatten()

		#f = interp2d(xx.flatten(), yy.flatten(), z, kind='cubic')


		nxx, nyy = np.meshgrid(np.arange(0,newImage.shape[1]), np.arange(0,newImage.shape[0]))

		nxx = nxx.flatten()
		nyy = nyy.flatten()

		print(nxx.shape)

		hfallten = newHomographyInverse.flatten()

		hpipe = np.zeros((nxx.shape[0], 9), dtype="float")

		ppipe =  np.zeros((nxx.shape[0], 9), dtype="float")


		hpipe[:,0] = hfallten[0]
		hpipe[:,1] = hfallten[1]
		hpipe[:,2] = hfallten[2]
		hpipe[:,3] = hfallten[3]
		hpipe[:,4] = hfallten[4]
		hpipe[:,5] = hfallten[5]
		hpipe[:,6] = hfallten[6]
		hpipe[:,7] = hfallten[7]
		hpipe[:,8] = hfallten[8]

		ppipe[:,0] = nxx;
		ppipe[:,1] = nyy;
		ppipe[:,2] = 1;

		ppipe[:,3] = nxx;
		ppipe[:,4] = nyy;
		ppipe[:,5] = 1;

		ppipe[:,6] = nxx;
		ppipe[:,7] = nyy;
		ppipe[:,8] = 1;

		multipy = hpipe * ppipe

		print(multipy.shape)

		newXCords = multipy[:,0] + multipy[:,1] + multipy[:,2]
		newYCords = multipy[:,3] + multipy[:,4] + multipy[:,5]
		newCCords = multipy[:,6] + multipy[:,7] + multipy[:,8]


		newXCords = newXCords/newCCords
		newYCords = newYCords/newCCords 

		oldCords = np.zeros((xx.shape[0],2), dtype="float")

		oldCords[:,0] = xx
		oldCords[:,1] = yy

		newValues = griddata(oldCords, z, (newXCords, newYCords), fill_value=0)
		maskValues = griddata(oldCords, (z*0+1), (newXCords, newYCords), fill_value=0)

		print(newValues.shape)

		transformedIm2 = newValues.reshape((newImage.shape[0],newImage.shape[1]))

		mask = maskValues.reshape((newImage.shape[0],newImage.shape[1]))

		temp = transformedIm2
	else:

		temp = cv2.warpPerspective(im2, newHomography, (totalX, totalY) )

	mask = cv2.warpPerspective(np.ones(im2.shape)*255, newHomography, (totalX, totalY) )

	#temp2 = cv2.warpPerspective(im2, H, (1000, 1000) )

	print("After homo shape")
	print(temp.shape)

	#Image.fromarray(temp.astype("uint8")).show()

	#newImage2 = newImage * 0;

	#newImage2[boundingBoxymin:boundingBoxymin+temp.shape[0], boundingBoxxmin:boundingBoxxmin + temp.shape[1]] = temp

	#Image.fromarray(temp.astype("uint8")).show()

	relativePointsY, relativePointsX = np.where(mask > 0)

	relativePointsXUp = relativePointsX #+  int(startingOffsetx)
	relativePointsYIp = relativePointsY #+ int(startingOffsety)


	newImage2 = newImage.copy()

	newImage2[relativePointsYIp, relativePointsXUp] = (newImage2[relativePointsYIp, relativePointsXUp] + temp[relativePointsY, relativePointsX])/2;

	#return (transformedIm2 + newImage)/2;

	mask = mask.astype("uint8")


	temp = np.clip(temp, 0, 255);
	newImage = np.clip(newImage, 0, 255);

	newTemp = np.zeros(newImage.shape, dtype="uint8")
	newMask = np.zeros(newImage.shape, dtype="uint8")


	newTemp[relativePointsYIp, relativePointsXUp] =  temp[relativePointsY, relativePointsX];
	newMask[relativePointsYIp, relativePointsXUp] =  mask[relativePointsY, relativePointsX];

	intersectionMap = newMask.astype("float") * sourceMask.astype("float");

	intersectionMap = (intersectionMap > 0)*255;


	semiAppended = newImage.copy()

	onlyIm2 = newMask.astype("float") -  intersectionMap.astype("float")

	idt = np.where(onlyIm2 > 0)

	onlyIm2 = (onlyIm2 > 0)*255;

	semiAppended[idt] = newTemp[idt]

	itt = np.where(intersectionMap > 0)

	#semiAppended[itt] =  0;

	temph = (intersectionMap > 0) * newTemp

	# Image.fromarray(temph.astype("uint8")).show()
	# Image.fromarray(newMask.astype("uint8")).show()
	# Image.fromarray(newImage2.astype("uint8")).show()
	# Image.fromarray(newMask.astype("uint8")).show()

	return semiAppended, newTemp, intersectionMap


def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()

  return signal.convolve2d(g_row, g_col, 'full')

def GaussianPDF_1D(mu, sigma, length):
  # create an array
  half_len = length / 2

  if np.remainder(length, 2) == 0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)

  ax = ax.reshape([-1, ax.size])
  denominator = sigma * np.sqrt(2 * np.pi)
  nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

  return nominator / denominator


def findDerivatives(I_gray):
  # TODO: your code here

  gaussianKernel = GaussianPDF_2D(GAUSSIAN_MEAN, GAUSSIAN_VAR, GAUSSIAN_SIZE_ROWS, GAUSSIAN_SIZE_COLS)

  filteredImage =  signal.convolve2d(I_gray, gaussianKernel, 'same')

  I_x_d = signal.convolve2d(filteredImage, SOBEL_X, 'same')

  I_y_d = signal.convolve2d(filteredImage, SOBEL_Y, 'same')

  I_m_d = np.power(np.power(I_x_d, 2) + np.power(I_y_d, 2), 1/2)

  I_d_d = -1.0 * I_y_d / (I_x_d + DIVIDE_EPSILON)

  return I_m_d, I_x_d, I_y_d, np.arctan(I_d_d)


def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def findOutputForColorImage(im1, im2, siftFlag, wrapFlag):


	square = rgb2gray(im1)
	squareShift = rgb2gray(im2)


	square.astype("float")
	squareShift.astype("float")

	square = imresize(square, 1.0)
	squareShift = imresize(squareShift, 1.0)
	
	H = getHomo(square, squareShift, siftFlag)

	redplane, redplaneD, redplaneA  = mergeImages(im1[:,:,0], im2[:,:,0], H, wrapFlag)
	greenplane, greenplaneD, greenplaneA  =  mergeImages(im1[:,:,1], im2[:,:,1], H, wrapFlag)
	blueplane, blueplaneD, blueplaneA = mergeImages(im1[:,:,2], im2[:,:,2], H, wrapFlag)


	newImageS = np.zeros((redplane.shape[0], redplane.shape[1], 3), dtype="uint8")
	newImageD = np.zeros((redplane.shape[0], redplane.shape[1], 3), dtype="uint8")
	newImageA = np.zeros((redplane.shape[0], redplane.shape[1], 3), dtype="uint8")

	newImageA[:,:,0] = redplaneA
	newImageA[:,:,1] = greenplaneA
	newImageA[:,:,2] = blueplaneA

	newImageS[:,:,0] = redplane
	newImageS[:,:,1] = greenplane
	newImageS[:,:,2] = blueplane

	newImageD[:,:,0] = redplaneD
	newImageD[:,:,1] = greenplaneD
	newImageD[:,:,2] = blueplaneD

	x1,x2,y1,y2 = bbox1(redplaneA)

	# x1 = x1 - 1;
	# x2 = x2 + 1; 
	# y1 = y1 - 1; 
	# y2 = y2 + 1;  



	newMask = newImageA[x1:x2+1, y1:y2+1,:]


	sImage = newImageS[x1:x2+1, y1:y2+1,:]
	dImage = newImageD[x1:x2+1, y1:y2+1,:]

	#Image.fromarray(newMask.astype("uint8")).show()
	#Image.fromarray(sImage.astype("uint8")).show()
	#Image.fromarray(dImage.astype("uint8")).show()

	mergedImage = cv2.seamlessClone(dImage, sImage, newMask.astype("uint8"), (int((sImage.shape[1])/2), int((sImage.shape[0])/2)), cv2.MIXED_CLONE)

	modifiedImage = newImageS.copy()

	modifiedImage[x1:x2+1, y1:y2+1,:] = mergedImage

	#mergedImage = np.clip(mergedImage, 0,255)

	#Image.fromarray(mergedImage.astype("uint8")).show()

	return modifiedImage


def getHomo(square, squareShift, siftFlag):

	x = None
	y = None
	xS = None
	yS = None
	fd = None
	fdS = None

	if not siftFlag:

		cmx = corner_detector(square)
		cmxS = corner_detector(squareShift)

		# Normalizing
		dst_norm = np.empty(cmx.shape, dtype=np.float32)
		cv2.normalize(cmx, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

		plt.imshow(cmx,cmap='jet')
		plt.show()

		dst_norm = np.empty(cmxS.shape, dtype=np.float32)
		cv2.normalize(cmxS, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

		
		plt.imshow(cmxS,cmap='jet')
		plt.show()
		#plt.plot(x,y,'ro')
		#plt.show()
		# corners_window = 'Corners detected'
		# cv2.namedWindow(corners_window)
		#cv2.imshow('h', dst_norm_scaled)

		print("anms computation")
		x,y,rmax = anms(cmx, Edge)
		xS,yS,rmaxS = anms(cmxS, Edge)
		print("anms done")

		#newSquare[y,x] = 255;
		#newSquareShift[yS, xS] = 255;


		#Image.fromarray(newSquare.astype("uint8")).show() 
		#Image.fromarray(newSquareShift.astype("uint8")).show() 


		fd =  implOne(square, x, y)
		fdS = implOne(squareShift, xS, yS)

		print("Descriptor Size")
		print(fd.shape)
	
	else:

		ppx, fd = computeSift(square)

		ppsx, fdS = computeSift(squareShift)

		x = ppx[:,0]
		y = ppx[:,1]

		xS = ppsx[:,0]
		yS = ppsx[:,1]

	implot = plt.imshow(square,cmap='gray')
	plt.plot(x,y,'ro',markersize = 2)
	plt.show()

	implot = plt.imshow(squareShift,cmap='gray')
	plt.plot(xS,yS,'ro',markersize=2)
	plt.show()

	mathingSet1 = feat_match(fdS, fd)
	mathingSet2 = feat_match(fd, fdS)


	map1 = {}

	smx = []
	smy = [] 
	dmx = [] 
	dmy = []

	for index in range(mathingSet1.shape[0]):
		if mathingSet1[index]!= -1:
			map1[(x[mathingSet1[index]], y[mathingSet1[index]])] = (xS[index], yS[index])


	mathes1to2 = []

	skpoints = []
	dkpoints = []

	count = 0;

	for index in range(mathingSet2.shape[0]):
		if mathingSet2[index]!= -1:
			newPoint = (xS[mathingSet2[index]], yS[mathingSet2[index]])
			currentPoint = (x[index], y[index])
			if currentPoint in map1 and map1[currentPoint] == newPoint:
				smx.append(x[index])
				smy.append(y[index])
				mathes1to2.append(cv2.DMatch(count, count, 1))
				count += 1

				dmx.append(xS[mathingSet2[index]])
				dmy.append(yS[mathingSet2[index]])

				skpoints.append(cv2.KeyPoint(x=x[index],y=y[index], _size=0))
				dkpoints.append(cv2.KeyPoint(x=xS[mathingSet2[index]], y=yS[mathingSet2[index]], _size=0))
			

	smx = np.array(smx)
	smy = np.array(smy)
	dmx = np.array(dmx)
	dmy = np.array(dmy)


	draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = None, flags = 2)

	src_pts = np.zeros((smx.shape[0],2), dtype="float32")
	dst_pts = np.zeros((smx.shape[0],2), dtype="float32")

	src_pts[:,0] = smx
	src_pts[:,1] = smy

	dst_pts[:,0] = dmx
	dst_pts[:,1] = dmy

	img3 = cv2.drawMatches(square,skpoints,squareShift,dkpoints,mathes1to2,None,**draw_params)

	Image.fromarray(img3.astype("uint8")).show()

	H, inliners = ransac_est_homography(dmx, dmy, smx, smy, SSD_THRES)

	indexesO = np.where(np.array(inliners)==0)
	print(indexesO)

	indexesI = np.where(np.array(inliners)==1)
	print(indexesI)

	sptsx = smx[indexesO]
	sptsy = smy[indexesO]
	dptsx = dmx[indexesO]
	dptsy = dmy[indexesO]

	ransacsqaure = square.copy()
	ransacsqaureshift = squareShift.copy()

	#ransacsqaure[sptsy,sptsx,:] = [0,0,255]
	#ransacsqaureshift[dptsy,dptsx,:] = [0,0,255]

	#ransacsqaure[smy[indexesI],smx[indexesI],:] = [255,0,0]
	#ransacsqaureshift[dmy[indexesI],dmx[indexesI],:] = [255,0,0]

	implot = plt.imshow(ransacsqaure,cmap='gray')
	plt.plot(sptsx,sptsy,'o',markersize=5)
	plt.plot(smx[indexesI],smy[indexesI],'ro',markersize=5)
	plt.show()

	implot = plt.imshow(ransacsqaureshift,cmap='gray')
	plt.plot(dptsx,dptsy,'o',markersize=5)
	plt.plot(dmx[indexesI],dmy[indexesI],'ro',markersize=5)
	plt.show()

	return H




def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox



def findOutputForColorImageArray(centerImage,orderBlending, siftFlag = True, wrapFlag = False, resize = 1.0):

	centerImage = imresize(centerImage, resize)

	currentImage = np.array(centerImage)

	orderBlending = [imresize(i, resize) for i in orderBlending]

	for destImage in orderBlending:
		currentImage = findOutputForColorImage(currentImage, np.array(destImage), siftFlag, wrapFlag)

	Image.fromarray(currentImage.astype("uint8")).show()


#def projectImage(image, cylinder):


































