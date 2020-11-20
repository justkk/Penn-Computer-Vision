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

outputpath = "verticalVideo.avi"

nr = 0;
nc = 100;


image = np.array(Image.open("testImage.jpg"))

imageSmall = imresize(image, 0.50)

I = np.array(imageSmall);

#I = np.transpose(I, (1, 0, 2));

energyMap = genEngMap(I)


Image.fromarray(energyMap).show()

finalImage, T, dataStore = wrapper(I, nr, nc)

imageList = []

size =  (I.shape[1],I.shape[0])

#Its vertical:

j = 0; 
while j < (nc +1):
	currentImage = dataStore[(0,j)][0]
	leftCompression = int(j/2); 
	rightCompression = j - int(j/2);
	updatedImage = helperUtils.paddChannelImage(currentImage, leftCompression, rightCompression)
	imageList.append(updatedImage[:, :, ::-1].copy())
	j += 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outputpath, fourcc, 5, size)
for i in range(len(imageList)):
	out.write(imageList[i].astype("uint8"))
out.release()