#########---------------------------------------------#########
'''
Script     = learnFaces.py
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
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator


from varConfig import *
#########---------------------------------------------#########


#########---------------------------------------------#########
def buildFaceImageGenerator():
    global trainDataSetpath, testDataSetpath

    # Build an image data generator objects
    imgGenTrain = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    imgGenTest = ImageDataGenerator(rescale = 1./255)
    # Build training and test image generator
    trainImgDataset = imgGenTrain.flow_from_directory(trainDataSetpath, target_size = imgTargetSize, batch_size = imageGenBatchSize, class_mode = 'categorical')
    testImgDataset = imgGenTest.flow_from_directory(testDataSetpath, target_size = imgTargetSize, batch_size = imageGenBatchSize, class_mode = 'categorical')
    class_indices = dict([(val, key) for key,val in trainImgDataset.class_indices.items()])
    return [{'TRAINING_DATA_GENERATOR':trainImgDataset, 'TESTING_DATA_GENERATOR':testImgDataset}, class_indices]



# class faceRecogModel(Model):
#     def __init__(self, inputShape):
#         super(faceRecogModel, self).__init__()

#         self.inputShape = inputShape
#         self.classCnt = len(glob(trainDataSetpath+'*'))
#         self.tfSeqModel = tf.keras.Sequential([
#             #layers.Input(shape=self.inputShape),
#             layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),

#             layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'), #),
#             layers.MaxPooling2D(), #(),

#             layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'), #),
#             layers.MaxPooling2D(), #(),

#             layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'), #),
#             layers.MaxPooling2D(), #(),

#             layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'), #),
#             layers.MaxPooling2D(), #(),

#             layers.Flatten(),
#             layers.Dense(self.classCnt, activation='softmax'),
#         ])
#     def call(self, input):
#         return self.tfSeqModel(input)

def buildSeqCNNModel(dataset):
    classCnt = len(glob(trainDataSetpath+'*'))


    # Build the model
    print(f" ##> Attempting to build a Squential model.\n");
    seqCNNModel = tf.keras.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='Layer1_Conv1'),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='Layer1_Conv2'),
            layers.MaxPooling2D(name='Layer1_MaxPool1'),

            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='Layer2_Conv1'),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='Layer2_Conv2'),
            layers.MaxPooling2D(name='Layer2_MaxPool1'),

            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='Layer3_Conv1'),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='Layer3_Conv2'),
            layers.MaxPooling2D(name='Layer3_MaxPool1'),

            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='Layer4_Conv1'),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='Layer4_Conv2'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='Layer4_Conv3'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='Layer4_Conv4'),
            layers.MaxPooling2D(name='Layer4_MaxPool1'),

            layers.Flatten(name='flatten1'),
            layers.Dense(64, activation='relu', name='Dense1'),
            layers.Dense(classCnt, activation='softmax', name='Dense2'),
        ])

    # Compile the model
    print(f" ##> Compiling the CNN model.\n");
    #seqCNNModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    seqCNNModel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    # Get the dataset
    trainImgDataset = dataset['TRAINING_DATA_GENERATOR']
    testImgDataset = dataset['TESTING_DATA_GENERATOR']

    # Fit the model to the data
    print(f" ##> Fitting the CNN model to the dataset.\n");
    seqCNNModelFit = seqCNNModel.fit(trainImgDataset, validation_data=testImgDataset, epochs=epochCount, steps_per_epoch=len(trainImgDataset), validation_steps=len(testImgDataset))
    seqCNNModel.summary()
    return (seqCNNModel, seqCNNModelFit)

def plotLossAcc(modelFit):
    # loss
    print(f" ##> Plotting the LOSS for the model.\n");
    plt.plot(modelFit.history['loss'], label='traing loss')
    plt.plot(modelFit.history['val_loss'], label='actual loss')
    plt.legend()
    #plt.show()
    plt.savefig('LossPlot')

    # accuracies
    print(f" ##> Plotting the ACCURACY for the model.\n");
    plt.plot(modelFit.history['accuracy'], label='train accuracy')
    plt.plot(modelFit.history['val_accuracy'], label='value accuracy')
    plt.legend()
    #plt.show()
    plt.savefig('AccuracyPlot')

def saveModel(model, modelName='learnFacesModel_cnn.h5'):
    print(f" ##> Saving the model {model} as {modelName}.\n");
    model.save(modelName)
#########---------------------------------------------#########


if __name__ == "__main__":
    testTrainDataset, class_indices = buildFaceImageGenerator()
    (model, modelFit) = buildSeqCNNModel(dataset = testTrainDataset)
    plotLossAcc(modelFit = modelFit)
    saveModel(model=model, modelName=modelNameToSave)
