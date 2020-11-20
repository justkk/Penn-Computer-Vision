import os
import sys
import cv2

def splitVideo(videoPath, folderPath):

    cap = cv2.VideoCapture(videoPath)

    videoName = videoPath.split("/")[-1].split(".")[0]

    os.system("rm -rf " + os.path.join(folderPath, videoName))
    os.system("mkdir " + os.path.join(folderPath, videoName))

    folderPath = os.path.join(folderPath, videoName)


    index = 0

    while True:

        success, sourceFrame = cap.read()

        if success == False:
            break

        fileName = os.path.join(folderPath, "IMG_" + str(index) + ".jpg")

        cv2.imwrite(fileName, sourceFrame)

        index += 1


videoPath = sys.argv[1]
outputPath = sys.argv[2]

splitVideo(videoPath, outputPath)

