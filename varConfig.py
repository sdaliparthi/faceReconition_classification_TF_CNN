# collectFace variables
haarFrontalFace    = 'haar_cascade_files/haarcascade_frontalface_default.xml' # Name and location of the XML file to detect frontal face
haarEyes           = 'haar_cascade_files/haarcascade_eye_tree_eyeglasses.xml' # Name and location of the XML file to detect eye(s)
scaleFactor        = 1.3 # Value of scaleFactor to be used in CascadeClassifier::detectMultiScale
minNeighbors       = 5 # Value of minNeighbors to be used in CascadeClassifier::detectMultiScale
camId              = 0 # Id of the camera to be used in the local machine
camImagesCnt       = 100 # No.of images to collect from camera for training purpose
imageStore         = './CamImagesStore/' # Name of the directory where images from camers should be stored

# learnFace variables
imgSize            = [224, 224, 3] # Size of the input for VGG19
imgTargetSize      = (224, 224) # Target size to be used for image generator
epochCount         = 5 # No.of epochs
imageGenBatchSize  = 16 # Batch saze to be used during image generator
trainDataSetpath   = imageStore + 'train/' # Location of the training dataset
testDataSetpath    = imageStore + 'test/' # Location of the testing dataset
modelNameToSave    = 'faceRecogModel_TF_CNN.h5' # Name to be used while saving the model

# recogFace variables
collectFacesDatSetUsingCam = False # Whether or not to collect the training and testing dataset
learnFacesUsingCNN         = False # Whether or not to learn and build the model based on testing and training data
