

import numpy as np

from PIL import Image

class HelperUtils():


	def shiftRowMatrix(self, row, shiftType, cValue):

		newMatrix = np.zeros((row.shape[0],1), dtype="float").flatten()

		newMatrix = newMatrix + cValue

		if shiftType == 1:
			newMatrix[1:] = row[:-1]
		else:
			newMatrix[:-1] = row[1:]

		return newMatrix


	def findRowMatrixMaxAndLabels(self, matrixList):

		labels = np.zeros((matrixList[0].shape[0], 1), dtype="float").flatten()
		maxValue = np.minimum.reduce(matrixList)
		for i in range(len(matrixList)):
			currentLabel = i + 1
			currentIndex = (matrixList[i] == maxValue)*currentLabel
			indexes = np.where(currentIndex!= 0)
			labels[indexes] = currentLabel

		return maxValue, labels



	def findVerticalSeam(self, energyMap):

		rows = energyMap.shape[0]
		cols = energyMap.shape[1]
		minMatrix = np.zeros((rows, cols), dtype="float")
		pathMatrix = np.zeros((rows, cols), dtype="float")
		minMatrix[0,:] = energyMap[0,:]

		for i in range(1,rows):
			parentRow = minMatrix[i-1,:]
			parentRowLeftShift = self.shiftRowMatrix(parentRow, -1, 1000000)
			parentRowRightShift = self.shiftRowMatrix(parentRow, 1, 1000000)
			minValues, labels = self.findRowMatrixMaxAndLabels([ parentRowRightShift, parentRow, parentRowLeftShift])
			minMatrix[i,:] = minValues + energyMap[i, :]
			pathMatrix[i,:] = labels - 2

		return minMatrix, pathMatrix


	def findPathAndUpdateMatrix(self, energyMap, pathMap ):

		rows, cols = energyMap.shape
		lastRow = energyMap[-1,:]
		minValueEnergy = np.min(lastRow)
		minIndexes = np.where(lastRow == minValueEnergy)
		minIndex = minIndexes[0][0]
		rowNumber = rows - 1
		colValue = minIndex
		valuesIndexes = []
		valuesIndexes.append(int(colValue))
		rowIndex = rows - 1


		while rowIndex > 0 :
			colValue = colValue + pathMap[rowIndex][int(colValue)]
			valuesIndexes.append(int(colValue))
			rowIndex -= 1

		valuesIndexes.reverse()


		return minValueEnergy, valuesIndexes


	def findUpdateMatrix(self, image, valuesIndexes):

		newImage = np.zeros((image.shape[0],image.shape[1]-1) , dtype="float")
		rows = image.shape[0]
		colMap = np.array(range(image.shape[1]))
		rowIndex = 0
		while rowIndex < rows:
			colMapUpdate = list(colMap + rowIndex*image.shape[1])
			currentRow = np.array(colMapUpdate[:valuesIndexes[rowIndex]] + colMapUpdate[valuesIndexes[rowIndex]+1:])
			newImage[rowIndex, :] = currentRow
			rowIndex += 1

		return newImage

	def makeImageFromMapImage(self, image, mapStore):

		mapStoreF = mapStore.flatten()
		values = image.flatten()[mapStoreF.astype("int")]
		return values.reshape((mapStore.shape[0],mapStore.shape[1]))

	# vertical
	def paddChannelImage(self, image, leftPad, rightPad):
		newImage = np.zeros((image.shape[0], image.shape[1] + leftPad + rightPad, image.shape[2]), dtype="float")
		for channel in range(image.shape[2]):
			currentChannel = image[:,:,channel]
			newImage[:,leftPad:image.shape[1]+leftPad,channel] = currentChannel

		return newImage

	# horz
	def paddChannelImageHorz(self, image, leftPad, rightPad):
		newImage = np.zeros((image.shape[0]+leftPad + rightPad, image.shape[1], image.shape[2]), dtype="float")
		for channel in range(image.shape[2]):
			currentChannel = image[:,:,channel]
			newImage[leftPad:image.shape[0]+leftPad,:,channel] = currentChannel

		return newImage

	def paddChannelImageTotal(self, image, heightCompression, widthCompression, fullSize):

		newImage = np.zeros(fullSize, dtype="float")

		leftPad = int(widthCompression/2);
		rightPad = widthCompression - leftPad;

		topPad = int(heightCompression/2)
		bottomPad = heightCompression - topPad

		for channel in range(image.shape[2]):
			currentChannel = image[:,:,channel]
			newImage[leftPad:image.shape[0]+leftPad, rightPad:image.shape[1]+rightPad,channel] = currentChannel

		return newImage


