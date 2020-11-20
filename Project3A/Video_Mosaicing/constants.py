import numpy as np 


SOBEL_X = 1 * np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
SOBEL_Y = 1 * np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


GAUSSIAN_MEAN = 0 
GAUSSIAN_VAR = 1.0

GAUSSIAN_SIZE_ROWS = 9
GAUSSIAN_SIZE_COLS = 9

DIVIDE_EPSILON = 0.00001


HIGHER_THRESHOLD = 0.045
LOWER_THRESHOLD = 0.01

STRIDE_ROW = 20
STRIDE_COL = 20

LINE_INFORMATION_LIMIT = 0.3


THRES = 1.0;

MAX = 100;


SSD_THRES = 4;

Edge = 800;
