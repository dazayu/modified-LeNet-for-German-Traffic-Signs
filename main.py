from lenet import LeNet
from keras.optimizers import SGD
from keras.utils import np_utils
#from keras import backend as K
import numpy as np
#import argparse
import scipy.io
#from matplotlib import pyplot as plt
#import h5py
import time
import xlsxwriter


##############################################
#   setting the parameters and hyperparameters
##############################################

dataPath = 'C:/Users/jiaoyi2/Documents/Clas a DataAug of GTS/Architecture2/'
# the rootpath of the folder
isEvaluation = False
# when True: only do the evaluation of the existing weights with the testDataset
# when false: train an architecture and test the architecture 
isExistingWeights = True
# when true: the code loads the weights and starts the training on the basis of 
#            the existing weights
# when false: the architecture will be trainied from scratch
listOfWeights = [
# elements are all string,
# the name of the weights data, is necessary when the isEvaluation is true
       'Epochs21to25_5_20190909105646'
        ]
groupOfEpochs = 5          
# One Group is five epochs
batchSize = 128
# the number of images in every batch
learningRate = 0.01
# the learning rate of the training processes

trainDataset = 'TrainOri'    
# String, name of the training set
testDataset = [                 
# The elements of the test data list should be all string
        'TestOri',
        'SeriallyAugmentedTestData',
        'TestRot',
        'TestBlur',
        'TestGammaContrast',
        'TestIntChange',
        'TestNois',
        'TestTrans',
        'AugTest'
        ]
sizeOfImg = [32,32,3]
# size of the input images [height,width,channel]
numOfClasses = 43
# number of the classes of the training/test dataset
dropoutRate = 0.5
# the rate of the dropout layer
regularizationParameter = 0.01


##############################################
# assistent Functions

def Print():
    print(" ")
    print("###########################")
    print(" ")
def Evaluation(num,iterGroup):
    nameOfTestData = testDataset[num]
    pathOfTestData = dataPath+nameOfTestData
    test_mat = scipy.io.loadmat(pathOfTestData)
    testData = test_mat["images"]
    testLabels = test_mat["labels"]
    testLabels = np_utils.to_categorical(testLabels, numOfClasses)

    print("[INFO] evaluating of the epochs",iterGroup*5+1,
          "to",(iterGroup+1)*5,"with the Test Dataset",nameOfTestData,"...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=batchSize, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    Print()       
    return loss,accuracy
##############################################
# the code of the training process

if isEvaluation == 0:
    nameOfTrainingData = dataPath+trainDataset
    train_mat = scipy.io.loadmat(nameOfTrainingData)
    trainData = train_mat["images"]
    trainLabels = train_mat["labels"]
    trainLabels = np_utils.to_categorical(trainLabels, numOfClasses)
    accuracyMatTrain = np.zeros((groupOfEpochs,5))
    accuracyMatTest = np.zeros((groupOfEpochs,len(testDataset)))
    lossMatTest = np.zeros((groupOfEpochs,len(testDataset)))
    
    for iterGroup in range (0,groupOfEpochs):
        if isExistingWeights == 0:
            weightDocName = ''
            print("[INFO] compiling model of the epochs",iterGroup*5+1,
                  "to",(iterGroup+1)*5,"...")
            opt = SGD(lr=learningRate)
            model = LeNet.build(numChannels=sizeOfImg[2],imgRows=sizeOfImg[0],
                                imgCols=sizeOfImg[1],numClasses=numOfClasses,
                                weightsPath=weightDocName if iterGroup > 0 else None,
                                dropoutR=dropoutRate,
                                regPara=regularizationParameter)
        else:
            assert len(listOfWeights)==1,"the listOfWeights should only have an \
            element when you want to start training from scratch"
            weightDocName = dataPath+listOfWeights[0]+'.hdf5'
            print("[INFO] compiling model of the epochs",iterGroup*5+1,
                      "to",(iterGroup+1)*5,"...")
            opt = SGD(lr=learningRate)
            model = LeNet.build(numChannels=sizeOfImg[2],imgRows=sizeOfImg[0],
                                    imgCols=sizeOfImg[1],numClasses=numOfClasses,
                                    weightsPath=weightDocName,
                                    dropoutR=dropoutRate,
                                    regPara=regularizationParameter)
        
        model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
                #if args["load_model"] < 0:
        print("[INFO] training of the epochs",iterGroup*5+1,
              "to",(iterGroup+1)*5,"...")
        history = model.fit(trainData,trainLabels,batch_size=batchSize,
                            initial_epoch=iterGroup*5,epochs=(iterGroup+1)*5,
                            verbose=1)
        # show the accuracy on the testing set
        accuracyMatTrain[iterGroup,:] = np.round(
                        np.asarray(history.history['acc']),4)
        Print()
        for iterOfTestDataset in range(0,len(testDataset)):
            (loss,acc) = Evaluation(iterOfTestDataset,iterGroup)     
            accuracyMatTest[iterGroup,iterOfTestDataset]=np.round(acc,4) 
            lossMatTest[iterGroup,iterOfTestDataset]=np.round(loss,4)
        print("[INFO] dumping weights to file ...")
        weightDocName = dataPath+trainDataset+"_Epochs{}to{}".format(iterGroup*5+1,(iterGroup+1)*5)+'_'+\
            time.strftime("_%Y%m%d%H%M%S",time.localtime())+'.hdf5'
        print("    ",weightDocName)
        model.save_weights(weightDocName, overwrite=True)
        Print()

##############################################
# the code of testing the existing weights

elif isEvaluation == 1: 
    assert listOfWeights != [], "forgot the name of weights"
    accuracyMatTest = np.zeros((len(listOfWeights),len(testDataset)))
    lossMatTest = np.zeros((len(listOfWeights),len(testDataset)))

    for iterOfWeights in range(0,len(listOfWeights)):
        weightDocName + dataPath+listOfWeights[iterOfWeights]
        print("[INFO] compiling model with the weight",listOfWeights[iterOfWeights],
              "...")
        opt = SGD(lr=learningRate)
        model = LeNet.build(numChannels=sizeOfImg[2],imgRows=sizeOfImg[0],
                            imgCols=sizeOfImg[1],numClasses=numOfClasses,
                            weightsPath=weightDocName,dropoutR=dropoutRate,
                            regPara=regularizationParameter)
        model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
        
        # evaluate the training dataset
        for iterOfTestDataset in range(0,len(testDataset)):
            (loss,acc) = Evaluation(iterOfTestDataset,iterGroup)     
            accuracyMatTest[iterOfWeights,iterOfTestDataset]=np.round(acc,4) 
            lossMatTest[iterOfWeights,iterOfTestDataset]=np.round(loss,4)

else:
    raise Exception('The isEvaluation should be boolean value!')

##############################################
# the code of saving as excel table
    
print(accuracyMatTrain)
print(accuracyMatTest)
print(lossMatTest)

workbookName = trainDataset+'_epochs_'+str(groupOfEpochs) \
               +time.strftime("_%Y%m%d%H%M%S",time.localtime())+'.xlsx'
workbook = xlsxwriter.Workbook(workbookName)
sheetTest = workbook.add_worksheet()
percentageFormal = workbook.add_format()
percentageFormal.set_num_format('0.00%')
decimalFormal = workbook.add_format()
decimalFormal.set_num_format('0.000')
sheetTest.set_column('B:J',13)
sheetTest.set_column('A:A',30)

for col in range(0,len(testDataset)):
    sheetTest.write(1,col+1,testDataset[col])
    sheetTest.write(1+groupOfEpochs+3,col+1,testDataset[col])
# the first row, the name of the test dataset

sheetTest.write(0,0,'the accuracy')

if isEvaluation == False:
    for row in range(0,groupOfEpochs):
        sheetTest.write(row+2,0,'the epoch {}'.format((row+1)*5))
        for col in range(0,len(testDataset)):
            sheetTest.write(row+2,col+1,accuracyMatTest[row,col],percentageFormal)
else:
    for row in range(0,len(listOfWeights)):
        sheetTest.write(row+2,0,'weight\'name'+listOfWeights[row])
        for col in range(0,len(testDataset)):
                sheetTest.write(row+2,col+1,accuracyMatTest[row,col],percentageFormal)

# the first column, the number of epochs or the name of the weight

sheetTest.write(3+groupOfEpochs,0,'the loss')
if isEvaluation == False:
    for row in range(0,groupOfEpochs):
        sheetTest.write(row+5+groupOfEpochs,0,'the epoch {}'.format((row+1)*5))
        for col in range(0,len(testDataset)):
            sheetTest.write(row+5+groupOfEpochs,col+1,lossMatTest[row,col],decimalFormal)
else:
    for row in range(0,len(listOfWeights)):
        sheetTest.write(row+5+groupOfEpochs,0,'weight\'name'+listOfWeights[row])
        for col in range(0,len(testDataset)):
                sheetTest.write(row+5+groupOfEpochs,col+1,lossMatTest[row,col],decimalFormal)
    
workbook.close()        
 
