
from finalproject.Code.OPEN_FACE import Open_Face

from finalproject.Code.VIDEO_MERGER import VideoMerger



srcVideo = "/Users/nikhilt/PycharmProjects/FS/finalproject/CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4"

targetVideo = "/Users/nikhilt/PycharmProjects/FS/finalproject/CIS581Project4PartCDatasets/Easy/MrRobot.mp4"

PATH = "shape_predictor_68_face_landmarks.dat"

instance = Open_Face(installationPath="/Users/nikhilt/Desktop/OpenFace", savePath="/Users/nikhilt/Desktop/playground/input",\
                     outputPath="/Users/nikhilt/Desktop/playground/output")



params = {}
params["VIDEO_WEIGHT"] = 0.5
params["TARGET_FRAME_INDEX"] = 47
params["FACE_MASK_TYPE"] = "FULL"
params["DLIB"] = False
params["EYE_TRACKING"] = False

if params["DLIB"]:
    params["EYE_TRACKING"] = False


vm = VideoMerger(srcVideo, targetVideo, instance, params)

st = ""
for key in params.keys():
    st += "__" + key + "_" +str(params[key])


prefix = targetVideo.split("/")[-1].split(".")[0] + "_to_" + srcVideo.split("/")[-1].split(".")[0]

vm.mergeVideo(params, prefix + "_test")
