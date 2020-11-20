
from finalproject.Code.PHOTO_MERGER import PHOTO_MERGER
from finalproject.Code.LAND_MARK_EXTRACTOR import Video_Land_Mark_Extractor
import cv2
from finalproject.Code.LAND_MARK_EXTRACTOR import HistoryMarking, getWeighedPoints, getFacePoints
import numpy as np


class VideoMerger(PHOTO_MERGER):

    def __init__(self, video1Path, video2Path, openFace, params):
        super(VideoMerger, self).__init__()
        self.sourcePath = video1Path
        self.targetPath = video2Path
        self.openFace = openFace
        self.params = params
        if video1Path is None or video2Path is None:
            return
        self.sourceVideoConverter = Video_Land_Mark_Extractor(self.openFace, self.sourcePath, self.params)
        #self.prnet = PRNET("/Users/nikhilt/PycharmProjects/FS/finalproject/PR/")
        #self.destVideoConverter = Video_Land_Mark_Extractor(self.openFace, self.sourcePath)

    def extractTargetFace(self, params):

        frameNumber = params["TARGET_FRAME_INDEX"]
        requiredImage = None
        vidcap = cv2.VideoCapture(self.targetPath)
        frameIndex = 0
        while True:
            success,targetImage = vidcap.read()
            if frameIndex == frameNumber:
                requiredImage = targetImage
                break

            frameIndex += 1

        return requiredImage


    def drawSourcePoints(self, image, points):

        for (x, y) in points:
            cv2.circle(image, (np.int(x), np.int(y)), 1, (0, 0, 255), -1)


        return image


    # def initialMerge(self, image1, image2, sourceBbox, targetBbox):
    #     return self.prnet.faceSwap(image1[:,:,::-1], image2[:,:,::-1], sourceBbox, targetBbox)[:,:,::-1]

    def bounding_box_naive(self, numpyPoints):

        points = []
        for i in range(numpyPoints.shape[0]):
            points.append((numpyPoints[i,0], numpyPoints[i,1]))

        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)

        return left, right, top, bottom


    def mergeVideo(self, params, saveFilePreix):

        saveFilePath = saveFilePreix + "_swap.avi"
        saveFilePath_LandMarks = saveFilePreix + "_land_marks.avi"

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        cap = cv2.VideoCapture(self.sourcePath)

        success, sourceFrame = cap.read()

        ROIMASK = np.zeros(sourceFrame.shape, dtype="uint8")

        ROIMASK[:, int(1*ROIMASK.shape[1]/4):int(3*ROIMASK.shape[1]/4) , :] = 1
        #ROIMASK[:,:,:] = 1

        sourceFrame = cv2.resize(sourceFrame,(sourceFrame.shape[1], sourceFrame.shape[0]))

        out = cv2.VideoWriter(saveFilePath,fourcc, cap.get(cv2.CAP_PROP_FPS), (sourceFrame.shape[1], sourceFrame.shape[0]), True)
        lnd = cv2.VideoWriter(saveFilePath_LandMarks, fourcc, cap.get(cv2.CAP_PROP_FPS), (sourceFrame.shape[1], sourceFrame.shape[0]), True)

        frameIndex = 0
        targetImage = self.extractTargetFace(params)
        targetImage = cv2.resize(targetImage,(sourceFrame.shape[1], sourceFrame.shape[0]))
        targetLandMarkPoints = self.openFace.getFacialLandMarks(targetImage, params)


        historyMarking = HistoryMarking([], [])

        oldFrame = None
        oldPoints = None

        while True:

            if success == False:
                break

            print("Extracting Frame: Index", frameIndex)

            sourceLandMarks = self.sourceVideoConverter.getLandMarkPoints(sourceFrame*ROIMASK, frameIndex, params)

            if oldPoints is not None:
                sourceLandMarks = getWeighedPoints(oldFrame, sourceFrame, oldPoints, sourceLandMarks)

            if sourceLandMarks is None:
                out.write(sourceFrame)
                frameIndex += 1
                success, sourceFrame = cap.read()
                continue

            # initialMerge = self.initialMerge(sourceFrame.copy(), targetImage.copy(),\
            #                                    self.bounding_box_naive(sourceLandMarks),\
            #                                    self.bounding_box_naive(targetLandMarkPoints))


            #Image.fromarray(initialMerge).show()

            # initialMergeLandMarkPoints = getFacePoints(initialMerge, False)
            #
            # if initialMergeLandMarkPoints is None:
            #      initialMergeLandMarkPoints = self.openFace.getFacialLandMarks(initialMerge, params)

            # morphedImage = self.morphImages(sourceFrame, initialMerge, sourceLandMarks.astype("int"), initialMergeLandMarkPoints.astype("int"), params)
            morphedImage = self.morphImages(sourceFrame, targetImage, sourceLandMarks.astype("int"), targetLandMarkPoints.astype("int"), params)

            #Image.fromarray(morphedImage[:,:,::-1]).show()

            historyMarking.addHistory(sourceLandMarks)

            morphedImage_land = self.drawSourcePoints(morphedImage.copy(), sourceLandMarks)

            frameIndex += 1

            out.write(morphedImage)
            lnd.write(morphedImage_land)

            oldPoints = sourceLandMarks.copy()

            oldFrame = sourceFrame.copy()

            success, sourceFrame = cap.read()


        print("Done")
        out.release()
        lnd.release()





