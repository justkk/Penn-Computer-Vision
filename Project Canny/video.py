import cv2, sys

from cannyEdge import cannyEdge
from challenges import getEdgeMapFromParams


lineMap = "N"
threshold = "G"
colorMap = "N"

videoPath = sys.argv[1]
vidcap = cv2.VideoCapture(videoPath)
outputpath = "challenge_results/" + videoPath.split("/")[-1].split(".")[0] + ".avi"
success,image = vidcap.read()
size =  (image.shape[1],image.shape[0])
count = 0
success = True
image_array = []
while success:
  edgeMap = getEdgeMapFromParams(image, threshold, lineMap, colorMap)
  image_array.append(edgeMap)
  #cv2.imwrite("frame%d.jpg" % count, edgeMap)     # save frame as JPEG file
  success,image = vidcap.read()
  print(count)
  count += 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outputpath, fourcc, 20, size)
for i in range(len(image_array)):
	out.write(image_array[i])
out.release()