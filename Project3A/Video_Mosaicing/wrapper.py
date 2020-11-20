

from PIL import Image


from helper import *


import sys




featArg = sys.argv[1]
wraper = sys.argv[2]

siftFlag = featArg == 'S'

wraper = wraper == 'I'

resize = float(sys.argv[3])


iarray = [Image.open("example-data/CMU2/medium01.JPG")]

centerImage = Image.open("example-data/CMU2/medium00.JPG");

findOutputForColorImageArray(centerImage, iarray, siftFlag, wraper, resize)

