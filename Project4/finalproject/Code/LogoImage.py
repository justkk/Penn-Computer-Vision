

from finalproject.Code.OPEN_FACE import Open_Face
from PIL import Image
import numpy as np
from finalproject.Code.VIDEO_MERGER import VideoMerger

params = {}
params["VIDEO_WEIGHT"] = 0.0
params["TARGET_FRAME_INDEX"] = 47
params["FACE_MASK_TYPE"] = "FULL"
params["DLIB"] = True
params["EYE_TRACKING"] = False

if params["DLIB"]:
    params["EYE_TRACKING"] = False



logoImagePath = "/Users/nikhilt/Desktop/Lordvoldemort.jpg"
nikhilImagePath = "/Users/nikhilt/Desktop/nikhil2.jpeg"

instance = Open_Face(installationPath="/Users/nikhilt/Desktop/OpenFace", savePath="/Users/nikhilt/Desktop/playground/input",\
                     outputPath="/Users/nikhilt/Desktop/playground/output")


logoImage = np.asarray(Image.open(logoImagePath).convert('RGB'))[:,:,::-1]
nikhilImage = np.asarray(Image.open(nikhilImagePath).convert('RGB'))[:,:,::-1]

nikhilRoi = np.zeros(logoImage.shape, dtype="uint8")

nikhilRoi[:,:,:] = 1

#nikhilRoi[:, int(1*nikhilRoi.shape[1]/3) : int(2*nikhilRoi.shape[1]/3), :] = 1

sourceFrame = logoImage * nikhilRoi

targetImage = nikhilImage.copy()
targetLandMarkPoints = instance.getFacialLandMarks(targetImage, params)
sourceLandMarks = instance.getFacialLandMarks(sourceFrame, params)

vm = VideoMerger(None, None, instance, params)

morphedImage = vm.morphImages(sourceFrame, targetImage, sourceLandMarks.astype("int"), targetLandMarkPoints.astype("int"), params)

Image.fromarray(morphedImage[:,:,::-1]).show()









