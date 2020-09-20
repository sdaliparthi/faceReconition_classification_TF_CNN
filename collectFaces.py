#########---------------------------------------------#########
'''
Script     = collectFaces.py
Author     = "Shashi Kanth Daliparthi"
License    = "GPL"
Version    = "1.0.0"
Maintainer = "Shashi Kanth Daliparthi"
E-mail     = "shashi.daliparthi@gmail.com"
Status     = "Production"
'''
#########---------------------------------------------#########


#########---------------------------------------------#########
# Import required modules
import cv2, os
import numpy as np
import matplotlib.pyplot


# Load the variables
from varConfig import *
#########---------------------------------------------#########


#########---------------------------------------------#########
# Create required instances
faceCascade = cv2.CascadeClassifier(haarFrontalFace)
eyesCascade = cv2.CascadeClassifier(haarEyes)
#########---------------------------------------------#########


#########---------------------------------------------#########
# To identify a face in the given image and mark it with a green boundary
def markFace(imgData):
    # Conert to gray scale
    #reqImgData = cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY) if imgData.shape[2] > 1 else imgData
    # Collect the locations of the face
    reqImgData = imgData
    reqDims = faceCascade.detectMultiScale(reqImgData, scaleFactor, minNeighbors)

    # Return if no face is detected
    if reqDims is ():
        print("##>  Unable to detect a face.\n")
        return None
    print("##> Successfully detected a face.\n")

    # Mark the eyes
    markEyes(reqImgData)

    # Crop the face
    for (xOrg, yOrg, width, height) in reqDims:
        faceCropData = reqImgData[yOrg:yOrg+height, xOrg:xOrg+width]
        cv2.rectangle(reqImgData, (xOrg, yOrg), (xOrg+width, yOrg+height), (0,255,0), 2)
    return faceCropData

# To identify eye(s) in the given image and mark it with a red boundary
def markEyes(imgData):
    # Conert to gray scale
    #reqImgData = cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY) if imgData.shape[2] > 1 else imgData
    reqImgData = imgData

    # Collect the locations of the eye(s)
    reqDims = eyesCascade.detectMultiScale(reqImgData, scaleFactor, minNeighbors)
    # Return if no face is detected
    if reqDims is ():
        print("##> Unable to detect a eye(s).\n")
        return None
    print("##> Successfully detects eye(s).\n")

    # Crop the eye(s)
    for (xOrg, yOrg, width, height) in reqDims:
        eyesCropData = reqImgData[yOrg:yOrg+height, xOrg:xOrg+width]
        cv2.rectangle(reqImgData, (xOrg, yOrg), (xOrg+width, yOrg+height), (255,0,0), 2)
    return eyesCropData

# To store images from cam on to local datastore
def storeFacesFromCam (count=100, path='./CamImagesStore'):
    # Initialize primary cam
    cv2.startWindowThread()
    camInst = cv2.VideoCapture(camId)

    # Collect and store the faces
    itr = 0
    while itr <= count:
        # Read image from cam
        flag, camImgData = camInst.read()
        reqCamImgData = camImgData.copy()

        # Identify a face in the captured image
        camImgFaceCrop = markFace(reqCamImgData)
        if camImgFaceCrop is not None:
            itr+=1
            imgName = path + str(itr) + '.jpg'
            print(" ##> Attempting to save the image %s.\n"%imgName)
            cv2.putText(camImgFaceCrop, str(itr), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imwrite(imgName, camImgData)
            cv2.imshow('--Cropped Face--', camImgFaceCrop)
        else:
            print(" ##> Unable to detect a face.\n")

        # Get out of loop when 'q' key is pressed or when required no.of images have been collected.
        if cv2.waitKey(1) == ord('q') or itr == count:
            print(" ##> Aborting the process of collecting and storing images from cam.\n")
            break

    print(" ##> Releasing the cam control.\n")
    camInst.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return

# To collect faces from the promary cam of the device and store them on the device
def colectFacesFrmCam (count = camImagesCnt, faceLabel=None):
    global imageStore, trainDataSetpath, testDataSetpath
    camTraingImagesCnt = count
    camTestingImagesCnt = int(0.20*count) # 20% of the training count

    # Collect Training Data set
    trainDataSetpath = imageStore + 'train/' + str(faceLabel) + '/'
    if not os.path.exists(trainDataSetpath): os.makedirs(trainDataSetpath)
    storeFacesFromCam(count=camTraingImagesCnt, path=trainDataSetpath)

    # Collect Testing Data set
    testDataSetpath = imageStore + 'test/' + str(faceLabel) + '/'
    if not os.path.exists(testDataSetpath): os.makedirs(testDataSetpath)
    storeFacesFromCam(count=camTestingImagesCnt, path=testDataSetpath)
#########---------------------------------------------#########



if __name__ == "__main__":
    while True:
        faceLabel = input("Enter face label (** Press ENTER to SKIP **) : ")
        if faceLabel == '': break
        colectFacesFrmCam(faceLabel = faceLabel)
        print(" ##> Successfully collected the faces from cam.\n")
