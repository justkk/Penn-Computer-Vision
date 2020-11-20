

from PIL import Image


from helper import *


import sys

from mymosaic import *



iarray = [Image.open("test_img/S1.jpg"), Image.open("test_img/S2.jpg"), Image.open("test_img/S3.jpg"), Image.open("test_img/S5.jpg"), Image.open("test_img/S6.jpg")]


mymosaic(iarray)