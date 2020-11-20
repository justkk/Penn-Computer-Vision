'''
  File name: helpers.py
  Author:
  Date created:
'''

'''
  File clarification:
    Helpers file that contributes the project
    You can design any helper function in this file to improve algorithm
'''
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from PIL import Image
from click_correspondences import *



samplePoints = np.array([[0,0],[10,0],[0,10],[10,10],[5,5]])


class MorphObject():

	def __init__(self, sourceImage, destImage, sourcePoints, destinationPoints ,avgPoints, triangulationStructure ,gamma):
		

		self.sourceImage = sourceImage
		self.destImage = destImage 
		self.imageStructure = np.zeros(self.sourceImage.shape, dtype="float")


		self.sourcePoints = sourcePoints		
		self.destinationPoints = destinationPoints

		self.avgPoints = avgPoints
		self.triangulationStructure = triangulationStructure

		self.gamma = gamma
		self.motionPoints = self.computeMotionPixels()
		
		
		self.dStructureTriangles = {}
		
		self.sourceStructureTriangles = {}
		self.destStructureTriangles = {}
		self.motionStructureTriangles = {}


		self.hMatrix = {}
		self.invHMatrix = {}

		self.sourceHMatrix = {}
		self.sourceInvHMatrix = {}

		self.destHMatrix = {}
		self.destInvHMatrix = {}

		self.motionHMatrix = {}
		self.motionInvHMatrix = {}


		self.buildStructureTriangles(self.triangulationStructure, self.motionPoints, self.hMatrix, self.invHMatrix, self.dStructureTriangles)
		
		self.buildStructureTriangles(self.triangulationStructure, self.sourcePoints, self.sourceHMatrix, self.sourceInvHMatrix, self.sourceStructureTriangles)

		self.buildStructureTriangles(self.triangulationStructure, self.destinationPoints, self.destHMatrix, self.destInvHMatrix, self.destStructureTriangles)

		self.buildStructureTriangles(self.triangulationStructure, self.motionPoints, self.motionHMatrix, self.motionInvHMatrix, self.motionStructureTriangles)






	def computeMotionPixels(self):
		motionPoints = self.gamma*self.sourcePoints + (1-self.gamma)*self.destinationPoints
		return motionPoints

	def buildStructureTriangles(self, triangulationStructure, points, hMatrix, invHMatrix, dStructureTriangles):
		triangleSystem = triangulationStructure.tri.simplices
		triangleNumber = triangleSystem.shape[0]

		for index in range(triangleNumber):

			cPoint1 = points[triangleSystem[index][0],:]
			cPoint2 = points[triangleSystem[index][1],:]
			cPoint3 = points[triangleSystem[index][2],:]
			# print(cPoint1, cPoint2, cPoint3)
			dStructure = Delaunay(np.array([cPoint1, cPoint2, cPoint3]))
			hMatrix[index] = TriangulationStructure.getMatrixForTriange( cPoint1, cPoint2, cPoint3)
			invHMatrix[index] =  np.linalg.inv(self.hMatrix[index])
			dStructureTriangles[index] = dStructure



	def labelTheTrianglesAndComputeBC(self):

		xCord, yCord = np.meshgrid(np.arange(self.imageStructure.shape[1]), np.arange(self.imageStructure.shape[0]))
		xCord = xCord.flatten()
		yCord = yCord.flatten()

		xCord = xCord.transpose()
		yCord = yCord.transpose()

		pointsZipped = np.zeros((xCord.shape[0], 2), dtype="float")
		pointsZipped[:,0] = xCord
		pointsZipped[:,1] = yCord

		pointsZipped3 = np.zeros((xCord.shape[0], 3), dtype="float")
		pointsZipped3[:,0] = xCord
		pointsZipped3[:,1] = yCord
		pointsZipped3[:,2] = 1

		# print(pointsZipped)

		labels = np.zeros((xCord.shape[0], 1), dtype="float").flatten()

		labels = (labels == 0) * -1

		firstPoints = np.zeros((xCord.shape[0], 2), dtype="float")
		secondPoints = np.zeros((xCord.shape[0], 2), dtype="float")
		thirdPoints = np.zeros((xCord.shape[0], 2), dtype="float")

		firstPointsS = np.zeros((xCord.shape[0], 2), dtype="float")
		secondPointsS = np.zeros((xCord.shape[0], 2), dtype="float")
		thirdPointsS = np.zeros((xCord.shape[0], 2), dtype="float")

		firstPointsD = np.zeros((xCord.shape[0], 2), dtype="float")
		secondPointsD = np.zeros((xCord.shape[0], 2), dtype="float")
		thirdPointsD = np.zeros((xCord.shape[0], 2), dtype="float")


		invMatrixSeries = np.zeros((3,3,xCord.shape[0]), dtype="float")

		bcArray = np.zeros((xCord.shape[0], 3), dtype="float")

		barycentricCordinates = np.zeros((xCord.shape[0], 3), dtype="float")

		for index in range(self.triangulationStructure.tri.simplices.shape[0]):

			replaceCords = np.where( labels == -1 )

			currentLabels = self.motionStructureTriangles[index].find_simplex(pointsZipped)

			itrLables = np.where(currentLabels >=0)

			firstPoints[itrLables] = self.motionPoints[self.triangulationStructure.tri.simplices[index][0]]
			secondPoints[itrLables] = self.motionPoints[self.triangulationStructure.tri.simplices[index][1]]
			thirdPoints[itrLables] = self.motionPoints[self.triangulationStructure.tri.simplices[index][2]]

			firstPointsS[itrLables] = self.sourcePoints[self.triangulationStructure.tri.simplices[index][0]]
			secondPointsS[itrLables] = self.sourcePoints[self.triangulationStructure.tri.simplices[index][1]]
			thirdPointsS[itrLables] = self.sourcePoints[self.triangulationStructure.tri.simplices[index][2]]

			firstPointsD[itrLables] = self.destinationPoints[self.triangulationStructure.tri.simplices[index][0]]
			secondPointsD[itrLables] = self.destinationPoints[self.triangulationStructure.tri.simplices[index][1]]
			thirdPointsD[itrLables] = self.destinationPoints[self.triangulationStructure.tri.simplices[index][2]]


			validPoints = np.zeros((itrLables[0].shape[0], 3), dtype="float")

			validPoints[:,[0,1]] = pointsZipped[itrLables][:,:]
			validPoints[:,2] = 1

			inverseRow1 = np.zeros((validPoints.shape[0],3), dtype="float")
			inverseRow2 = np.zeros((validPoints.shape[0],3), dtype="float")
			inverseRow3 = np.zeros((validPoints.shape[0],3), dtype="float")

			inverseRow1[np.arange(itrLables[0].shape[0]),:] = self.motionInvHMatrix[index][0,:]
			inverseRow2[np.arange(itrLables[0].shape[0]),:] = self.motionInvHMatrix[index][1,:]
			inverseRow3[np.arange(itrLables[0].shape[0]),:] = self.motionInvHMatrix[index][2,:]

			p1 = inverseRow1 * pointsZipped3[itrLables]
			p2 = inverseRow2 * pointsZipped3[itrLables]
			p3 = inverseRow3 * pointsZipped3[itrLables]

			p1 = np.sum(p1,1)
			p2 = np.sum(p2,1)
			p3 = np.sum(p3,1)

			barycentricCordinates[list(itrLables[0]),0] = p1
			barycentricCordinates[list(itrLables[0]),1] = p2
			barycentricCordinates[list(itrLables[0]),2] = p3


			# validPoints = np.transpose(validPoints)

			currentLabels = (currentLabels >= 0) * (index+1)

			currentLabels = currentLabels - 1;

			labels[replaceCords] = currentLabels[replaceCords]


		labelsMatrix = np.reshape(labels, (self.imageStructure.shape[1], self.imageStructure.shape[0]))

		#labelsMatrix = labelsMatrix.transpose()

		# print(labelsMatrix)

		# vecCord = np.zeros((xCord.shape[0], 3), dtype="float")
		# vecCord[:,0] = xCord
		# vecCord[:,1] = yCord
		# vecCord[:,2] = 1

		# pointsList = list(vecCord)

		# barycentricCordinates = [];

		# print(self.motionPoints[labels])

		# print("TIME3")

		# for index in range(vecCord.shape[0]):
		# 	matrix = self.motionInvHMatrix[labels[index]] 
		# 	point = np.transpose(vecCord[index, :])
		# 	barycentricCordinates.append(np.matmul(matrix, point))

		# print("TIME4")


		return list(labels), barycentricCordinates, pointsZipped, firstPoints, secondPoints, thirdPoints, firstPointsS, secondPointsS, thirdPointsS,firstPointsD, secondPointsD, thirdPointsD


	def getCordinate(self,point1, point2, point3, bc):

		x = point1[0] * bc[0] + point2[0]* bc[1] + point3[0] * bc[2]
		y = point1[1] * bc[0] + point2[1]* bc[1] + point3[1] * bc[2]
		z = 1 * bc[0] + 1* bc[1] + 1 * bc[2] 

		return np.array([x/z , y/z]) 


	def mapPixels(self,sourceImage, sourcePoints, sourceHMatrix, sourceInvHMatrix , targetPoints, targetHMatrix, targetInvMatrix, targetBc , targetLabels, triangleStructure):

		fractionPixels = [];
		fractionValue = [];

		function = interpolate.RectBivariateSpline(np.arange(sourceImage.shape[1]), np.arange(sourceImage.shape[0]), np.transpose(sourceImage))

		for index in range(len(targetLabels)):

			triangle = triangleStructure.tri.simplices[targetLabels[index], :]

			point1 = sourcePoints[triangle[0], :]
			point2 = sourcePoints[triangle[1], :]
			point3 = sourcePoints[triangle[2], :]
			bc = targetBc[index]

			sourcePixelLocation = self.getCordinate(point1, point2, point3, bc)

			fractionPixels.append(sourcePixelLocation)

			fractionValue.append(function(sourcePixelLocation[0], sourcePixelLocation[1]))
		
		targetImage = np.reshape(np.array(fractionValue), (sourceImage.shape[0], sourceImage.shape[1]))
		return targetImage


	def mapPixelsVec(self,sourceImage, sourcePoints, sourceHMatrix, sourceInvHMatrix , targetPoints, targetHMatrix, targetInvMatrix, bc , targetLabels, triangleStructure, firstPoint, secondPoint, thirdPoint):

		fractionPixels = [];
		fractionValue = [];

		function = interpolate.RectBivariateSpline(np.arange(sourceImage.shape[1]), np.arange(sourceImage.shape[0]), np.transpose(sourceImage))

		mixedX = bc[:,0]* firstPoint[:,0] + bc[:,1]* secondPoint[:,0] + bc[:,2]* thirdPoint[:,0]
		mixedY = bc[:,0]* firstPoint[:,1] + bc[:,1]* secondPoint[:,1] + bc[:,2]* thirdPoint[:,1]  

		mixedZ = bc[:,0] + bc[:,1] + bc[:,2]

		mixedX = mixedX/mixedZ
		mixedY = mixedY/mixedZ

		fractionValues = function.ev(mixedX, mixedY)
		targetImage = np.reshape(fractionValues, (sourceImage.shape[0], sourceImage.shape[1]))
		return targetImage





		# for index in range(len(targetLabels)):

		# 	triangle = triangleStructure.tri.simplices[targetLabels[index], :]

		# 	point1 = sourcePoints[triangle[0], :]
		# 	point2 = sourcePoints[triangle[1], :]
		# 	point3 = sourcePoints[triangle[2], :]
		# 	bc = targetBc[index]

		# 	sourcePixelLocation = self.getCordinate(point1, point2, point3, bc)

		# 	fractionPixels.append(sourcePixelLocation)

		# 	fractionValue.append(function(sourcePixelLocation[0], sourcePixelLocation[1]))
		
		# targetImage = np.reshape(np.array(fractionValue), (sourceImage.shape[0], sourceImage.shape[1]))
		# return targetImage




class TriangulationStructure(object):
	"""docstring for TriangulationStructure"""
	def __init__(self, points):
		self.points = points
		self.tri = self.findTriangulationOfPoints()
		self.triangleHMatrix = {}
		self.triangleInvHMatrix = {}
		self.buildTriangleHMatrix()


	def buildTriangleHMatrix(self):
		triangleSystem = self.tri.simplices
		triangleNumber = triangleSystem.shape[1]
		for index in range(triangleNumber):
			point1 = self.points[triangleSystem[index][0],:]
			point2 = self.points[triangleSystem[index][1],:]
			point3 = self.points[triangleSystem[index][2],:]
			hMatrix = TriangulationStructure.getMatrixForTriange(point1, point2, point3)
			hMatrixInverse =  np.linalg.inv(hMatrix)
			self.triangleHMatrix[index] = hMatrix
			self.triangleInvHMatrix[index] = hMatrixInverse





	@staticmethod
	def getMatrixForTriange(point1, point2, point3):
		hMatrix = np.ones((3,3)) * 1.0
		hMatrix[0:2,0] = np.transpose(point1)
		hMatrix[0:2,1] = np.transpose(point2)
		hMatrix[0:2,2] = np.transpose(point3)
		return hMatrix



	def findTriangulationOfPoints(self):
		tri = Delaunay(self.points)
		# print(tri.simplices)
		return tri
	def showTriangulation(self):
		plt.triplot(self.points[:,0], self.points[:,1], self.tri.simplices.copy())
		plt.plot(self.points[:,0],self. points[:,1], 'o')
		plt.show()



def processChannel(sourceImage, destImage, sourcePoints, destinationPoints, avgPoints, triStructure, gamma, dissolve):

	a = MorphObject(sourceImage, destImage, sourcePoints, destinationPoints, avgPoints, triStructure, gamma)

	labels, bc, points, firstPoints, secondPoints, thirdPoints, firstPointsS, secondPointsS, thirdPointsS,firstPointsD, secondPointsD, thirdPointsD = a.labelTheTrianglesAndComputeBC()

	#targetImage1 = a.mapPixels(a.sourceImage, a.sourcePoints, a.sourceHMatrix, a.sourceInvHMatrix , a.motionPoints, a.motionHMatrix, a.motionInvHMatrix, bc , labels, triStructure)
	
	#targetImage2 = a.mapPixels(a.destImage, a.destinationPoints, a.destHMatrix, a.destInvHMatrix , a.motionPoints, a.motionHMatrix, a.motionInvHMatrix, bc , labels, triStructure)

	targetImage1 = a.mapPixelsVec(a.sourceImage, a.sourcePoints, a.sourceHMatrix, a.sourceInvHMatrix , a.motionPoints, a.motionHMatrix, a.motionInvHMatrix, bc , labels, triStructure, firstPointsS, secondPointsS, thirdPointsS)
	targetImage2 = a.mapPixelsVec(a.destImage, a.destinationPoints, a.destHMatrix, a.destInvHMatrix , a.motionPoints, a.motionHMatrix, a.motionInvHMatrix, bc , labels, triStructure, firstPointsD, secondPointsD, thirdPointsD)

	targetImage =  targetImage1 * dissolve + targetImage2 * (1-dissolve)
	#targetImage =   targetImage2 * (1-dissolve)

	targetImage = np.round(targetImage)

	np.clip(targetImage, 0, 255, out=targetImage)

	return targetImage





# sourceImage = np.array(Image.open("test.png"))

# destImage = np.array(Image.open("test.png"))

# sourceImage = sourceImage[:,:,1]


# # sourcePoints = np.array([[0,0],[0,256],[256,0],[256,256],[128,128]])

# # destinationPoints = np.array([[0,0],[0,256],[256,0],[256,256],[128,2]])

# #image = np.reshape(np.arange(121), (11,11))

# gamma = 0.5

# dissolve = 0.5


# #Image.fromarray(sourceImage.astype("uint8")).show()

# sourcePoints, destinationPoints = click_correspondences(sourceImage, sourceImage)

# avgPoints = (sourcePoints + destinationPoints)/2

# triStructure = TriangulationStructure(avgPoints)

# processChannel(sourceImage, destImage, sourcePoints, destinationPoints, avgPoints, triStructure, gamma, dissolve)

