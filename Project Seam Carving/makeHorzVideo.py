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

outputpath = "horzVideo.avi"

nr = 100;
nc = 0;


image = np.array(Image.open("testImage.jpg"))

imageSmall = imresize(image, 0.50)

I = np.array(imageSmall);

finalImage, T, dataStore = wrapper(I, nr, nc)

imageList = []

size =  (imageSmall.shape[1],imageSmall.shape[0])

#Its vertical:

j = 0; 
while j < (nr +1):
	currentImage = dataStore[(j,0)][0]
	topCompression = int(j/2); 
	bottomCompression = j - int(j/2);
	updatedImage = helperUtils.paddChannelImageHorz(currentImage, topCompression, bottomCompression)
	imageList.append(updatedImage[:, :, ::-1].copy())
	j += 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outputpath, fourcc, 5, size)
for i in range(len(imageList)):
	out.write(imageList[i].astype("uint8"))
out.release()