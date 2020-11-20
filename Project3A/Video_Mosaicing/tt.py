#a = [[ 91,44],[178,133],[120,63],[ 88,29]]
#b = [[ 28,49],[178,133],[ 61,69],[ 21,33]]




import cv2

import numpy as np

from PIL import Image

from scipy.misc import imresize


square = np.array(Image.open("test_img/1M.jpg"))

square = imresize(square, 0.125) 

from est_homography import *



xx = square.shape[1]
yy = square.shape[0]

x = np.array([0,0,xx,xx]).astype("float32")
y = np.array([0,yy,0,yy]).astype("float32")

X = np.array([0,0,xx,xx]).astype("float32")
Y = np.array([yy*1.0/4,yy*3.0/4,-yy*1.0/4,yy*5.0/4]).astype("float32")

H = est_homography(x,y,X,Y)

print(H)



newImage = cv2.warpPerspective(square, H, (square.shape[1],square.shape[0]))

print(newImage.shape)
print(square.shape)

Image.fromarray(newImage.astype("uint8")).show()
Image.fromarray(square.astype("uint8")).show()