

from PIL import Image

import os

import numpy as np

import uuid

class Open_Face():

    def __init__(self, installationPath="", outputPath = "", savePath = ""):
        self.path = ""
        self.savePath = savePath
        self.installationPath = installationPath
        self.outputPath = outputPath

    def saveFile(self, image):
        fileName = str(uuid.uuid4()) + ".jpg"

        filePath = os.path.join(self.savePath, fileName)
        image = image.astype("uint8")
        pilObj = Image.fromarray(image[:,:,::-1])
        pilObj.save(filePath)
        return fileName, filePath

    def executeCommandPhoto(self, fileName, filePath):

        func = self.installationPath + "/build/bin/FaceLandmarkImg"

        folderPath = os.path.join(self.outputPath, fileName.split(".")[0])
        os.system("rm -rf " + folderPath)
        os.system("mkdir " + folderPath)
        command = func + " -f "  + filePath  + " -out_dir "  + folderPath
        os.system(command)
        print("Completed executing landmarks for", fileName)

        return folderPath

    def executeCommandVideo(self, videoPath):

        func = self.installationPath + "/build/bin/FaceLandmarkVidMulti"
        folderPath =  os.path.join(self.outputPath, str(uuid.uuid4()))
        os.system("rm -rf " + folderPath)
        os.system("mkdir " + folderPath)
        command = func + " -f "  + videoPath  + " -out_dir "  + folderPath
        os.system(command)
        print("Completed executing landmarks for video", videoPath)
        return folderPath


    def parseOutputFileVideo(self, fileName, folderPath, params):

        csvFileLocation = os.path.join(folderPath, fileName.split(".")[0] + ".csv")

        file = open(csvFileLocation, "r")
        lines = file.readlines()

        headers = lines[0].split(",")[299:299+68+68]
        print("headers")


        frameInformation = {}

        for index in range(1, len(lines)):

            line = lines[index].strip().split(",")

            frameIndex = int(line[0]) - 1
            faceIndex = int(line[1])

            faceInformation  = line[299:299+68+68]

            xcords = faceInformation[0:68]
            ycords = faceInformation[68:]

            headers = lines[0].split(",")[10+3:122+3]

            eyeShape = lines[1].split(",")[10+3:122+3]

            eyeShape = eyeShape[0:20] + eyeShape[28:48] + eyeShape[56:76] + eyeShape[84: 104]

            if params["EYE_TRACKING"]:
                pointArray = np.zeros((68 + int(len(eyeShape)/2), 2))
            else:
                pointArray = np.zeros((68, 2))

            pointArray[0:68, 0] = xcords
            pointArray[0:68, 1] = ycords

            if params["EYE_TRACKING"]:
                pointArray[68:, 0] = eyeShape[0:int(len(eyeShape)/2)]
                pointArray[68:, 1] = eyeShape[int(len(eyeShape)/2):]


            if frameIndex not in frameInformation:
                frameInformation[frameIndex] = {}


            frameInformation[frameIndex][faceIndex] = {}
            frameInformation[frameIndex][faceIndex]["pointArray"] = pointArray

        return frameInformation


    def parseOutputFilePhoto(self, fileName, folderPath, params):

        csvFileLocation = os.path.join(folderPath, fileName.split(".")[0] + ".csv")

        file = open(csvFileLocation, "r")
        lines = file.readlines()

        headers = lines[0].split(",")[296:296+68+68]
        print("headers")
        face1  = lines[1].split(",")[296:296+68+68]

        xcords = face1[0:68]
        ycords = face1[68:]

        headers = lines[0].split(",")[10:122]

        headers = headers[0:20] + headers[28:48] + headers[56:76] + headers[84: 104]

        eyeShape = lines[1].split(",")[10:122]

        eyeShape = eyeShape[0:20] + eyeShape[28:48] + eyeShape[56:76] + eyeShape[84: 104]

        if params["EYE_TRACKING"]:
            pointArray = np.zeros((68 + int(len(eyeShape)/2), 2))
        else:
            pointArray = np.zeros((68, 2))

        pointArray[0:68, 0] = xcords
        pointArray[0:68, 1] = ycords

        if params["EYE_TRACKING"]:
            pointArray[68:, 0] = eyeShape[0:int(len(eyeShape)/2)]
            pointArray[68:, 1] = eyeShape[int(len(eyeShape)/2):]


        return pointArray


    def getFacialLandMarks(self, image, params):

        image = image.copy()

        fileName, filePath = self.saveFile(image)
        folderPath = self.executeCommandPhoto(fileName, filePath)
        points = self.parseOutputFilePhoto(fileName, folderPath, params)
        return points


    def getFacialLandMarksInVideo(self, videoPath, params):
        fileName = videoPath.split("/")[-1].split(".")[0]
        folderPath = self.executeCommandVideo(videoPath)
        return self.parseOutputFileVideo(fileName, folderPath, params)

