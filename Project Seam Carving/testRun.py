

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


#array = np.array([[1,2,3],[8,4,2],[1,5,3]])

#I = np.zeros((array.shape[0], array.shape[1],1), dtype="float")

#I[:,:,0] = array

image = np.array(Image.open("testImage.jpg"))

imageSmall = imresize(image, 0.5)

energyMap = genEngMap(imageSmall)

I = np.array(imageSmall);


nr = 10
nc = 10

output = carv(I, nr, nc)

Image.fromarray(energyMap.astype("uint8")).show()
Image.fromarray(imageSmall.astype("uint8")).show()
Image.fromarray(output[0].astype("uint8")).show()

#print(output[0][:,:,0], output[1])