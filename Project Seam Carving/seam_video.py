from helpers import *

import numpy as np


from constants import *

from cumMinEngVer import *

from cumMinEngHor import *

from rmHorSeam import *

from rmVerSeam import *

from carv import *

from PIL import Image

from scipy.misc import imresize

import cv2, sys



#array = np.array([[1,2,3],[8,4,2],[1,5,3]])

#I = np.zeros((array.shape[0], array.shape[1],1), dtype="float")

#I[:,:,0] = array

# image = np.array(Image.open("testImage.jpg"))

# imageSmall = imresize(image, 0.50)

# I = np.array(imageSmall);

# finalImage, T, dataStore = wrapper(I, 0, compressionLength)

# imageList = []

# size =  (imageSmall.shape[1],imageSmall.shape[0])

# #Its vertical:

# j = 0; 
# while j < (compressionLength +1):
# 	currentImage = dataStore[(0,j)][0]
# 	leftCompression = int(j/2); 
# 	rightCompression = j - int(j/2);
# 	updatedImage = helperUtils.paddChannelImage(currentImage, leftCompression, rightCompression)
# 	imageList.append(updatedImage[:, :, ::-1].copy())
# 	j += 1

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(outputpath, fourcc, 5, size)
# for i in range(len(imageList)):
# 	out.write(imageList[i].astype("uint8"))
# out.release()

compressionLength = 50;
nr = 0; 
nc = compressionLength;
def parallel_function(image):
	finalImage, T, dataStore = wrapper(image, nr, nc)
	output = finalImage[:, :, ::-1].copy()
	print(image.shape, finalImage.shape, output.shape)
	return output



videoPath = sys.argv[1]
vidcap = cv2.VideoCapture(videoPath)
outputpath = "challenge_results_" + videoPath.split("/")[-1].split(".")[0] + ".avi"
success,image = vidcap.read()
count = 0
success = True

orig_image_array = []
image_array = []

size =  None 

while success:
	#orig_image_array.append(image)
	output = parallel_function(image)
	image_array.append(output.copy())
	print(image_array[0].shape)
	success,image = vidcap.read()
	print(count)
	print(output.shape)
	count += 1

finalImage = image_array[0]
size = (finalImage.shape[1],finalImage.shape[0])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outputpath, fourcc, 20, size)
for i in range(len(image_array)):
	out.write(image_array[i].astype("uint8"))
out.release()