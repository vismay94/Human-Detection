from os import listdir
from os.path import isfile, join
import random
import numpy as np
import cv2
import math
import prewitt as prewit
import cellAndBin as cab
import createHoGDescriptor as hogDesc
import neuralNetwork as nn


def main():
    imageData = []
    imageResult = []
    mypathPos='/users/vismay/desktop/Human/Train_Positive/'
    onlyfilesPos = [ f for f in listdir(mypathPos) if isfile(join(mypathPos,f)) ]
    
    imagesPos = np.empty(len(onlyfilesPos), dtype=object)
    for n in range(0, len(onlyfilesPos)):
        imagesPos[n] = cv2.imread( join(mypathPos,onlyfilesPos[n]) )

    # img = cv2.imread('/users/vismay/desktop/Human/Test_Positive/image1.bmp',cv2.IMREAD_COLOR)
    for i in range(len(imagesPos)):
        name= str(onlyfilesPos[i])
        img = imagesPos[i]
        size = img.shape
        rows = size[0]
        cols = size[1]
        grayScaleImage = np.zeros((img.shape[0],img.shape[1]))
        for i in range(0,rows):
            for j in range(0,cols):
                red = img[i,j,2]
                blue = img[i,j,0]
                green = img[i,j,1]
                pixelVal = round(0.299*red + 0.587*green + 0.114 *blue)
                grayScaleImage[i,j] = pixelVal
                
        # cv2.imwrite('grayScaleImage.bmp',grayScaleImage)
        
        prewitImage, prewitGxImage, prewitGyImage, gradientAngle = prewit.prewittOperation(grayScaleImage,rows,cols)
        # cv2.imwrite(name,prewitImage)
        histogram = cab.createCellHistogram(prewitImage, gradientAngle)

        HoGDescriptor = hogDesc.getHogDescriptor(histogram[0], histogram[1])

        if(name == "crop001278a.bmp"):
            np.savetxt("crop001278a.txt", HoGDescriptor, newline='\r\n')

        imageData.append(HoGDescriptor)
        imageResult.append(1)

    print("Descriptor shape after Positive Reading: ", len(imageData[0]))

    mypathNeg='/users/vismay/desktop/Human/Train_Negative/'
    onlyfilesNeg = [ f for f in listdir(mypathNeg) if isfile(join(mypathNeg,f)) ]
    imagesNeg = np.empty(len(onlyfilesNeg), dtype=object)
    for n in range(0, len(onlyfilesNeg)):
        imagesNeg[n] = cv2.imread( join(mypathNeg,onlyfilesNeg[n]) )    

    for i in range(len(imagesPos)):
        img = imagesNeg[i]
        size = img.shape
        rows = size[0]
        cols = size[1]
        grayScaleImage = np.zeros((img.shape[0],img.shape[1]))
        for i in range(0,rows):
            for j in range(0,cols):
                red = img[i,j,2]
                blue = img[i,j,0]
                green = img[i,j,1]
                pixelVal = round(0.299*red + 0.587*green + 0.114 *blue)
                grayScaleImage[i,j] = pixelVal
                
        # cv2.imwrite('grayScaleImage.bmp',grayScaleImage)
        
        prewitImage, prewitGxImage, prewitGyImage, gradientAngle = prewit.prewittOperation(grayScaleImage,rows,cols)

        histogram = cab.createCellHistogram(prewitImage, gradientAngle)

        HoGDescriptor = hogDesc.getHogDescriptor(histogram[0], histogram[1])
        imageData.append(HoGDescriptor)
        imageResult.append(0)
    
    # Shuffling Data randomly for better result.
    c = list(zip(imageData, imageResult))
    random.shuffle(c)
    imageData, imageResult = zip(*c)

    # This Loop will use different number of Neurons in Hidden Layer, We will call a function which will get trained
    # and will return dictionary, which will be used to predict other images.
    for hidden_neurons in [250,500,1000]:
        print("Training Started for Hidden Neurons", hidden_neurons)
        dictionary = nn.createNeuralNetwork(imageData,imageResult,hidden_neurons)
        print("Saving Data")
        nn.saveDictionary(dictionary,"valuesOfNeurons"+str(hidden_neurons))
        print("--------")

    # We will read the positive test images and Create HoG for each.
    testImageData = []
    testImageResult = []
    testImagePathPos='/users/vismay/desktop/CvImages/Test_Positive'
    testFilesPos = [ f for f in listdir(testImagePathPos) if isfile(join(testImagePathPos,f)) ]
    testImagesPos = np.empty(len(testFilesPos), dtype=object)
    for n in range(0, len(testFilesPos)):
        testImagesPos[n] = cv2.imread( join(testImagePathPos,testFilesPos[n]) )

    for i in range(len(testImagesPos)):
        name= str(testFilesPos[i])
        imgs = testImagesPos[i]
        size = imgs.shape
        rows = size[0]
        cols = size[1]
        grayScaleImage = np.zeros((imgs.shape[0],imgs.shape[1]))
        for i in range(0,rows):
            for j in range(0,cols):
                red = imgs[i,j,2]
                blue = imgs[i,j,0]
                green = imgs[i,j,1]
                pixelVal = round(0.299*red + 0.587*green + 0.114 *blue)
                grayScaleImage[i,j] = pixelVal
                
        # cv2.imwrite('grayScaleImage.bmp',grayScaleImage)
        
        prewitImage, prewitGxImage, prewitGyImage, gradientAngle = prewit.prewittOperation(grayScaleImage,rows,cols)
        cv2.imwrite(name,prewitImage)
        histogram = cab.createCellHistogram(prewitImage, gradientAngle)

        HoGDescriptor = hogDesc.getHogDescriptor(histogram[0], histogram[1])
        
        if(name == "crop001045b.bmp"):
            print("Saving for the name",name)
            np.savetxt("crop001045b.txt", HoGDescriptor, newline='\r\n')
            print("Saved Desriptior")

        testImageData.append(HoGDescriptor)
        testImageResult.append(1)


    # We will read the Negative test images and Create HoG for each.
    testImagePathNeg='/users/vismay/desktop/CvImages/Test_Neg'
    testFilesNeg = [ f for f in listdir(testImagePathNeg) if isfile(join(testImagePathNeg,f)) ]
    testImagesNeg = np.empty(len(testFilesNeg), dtype=object)
    for n in range(0, len(testFilesNeg)):
        testImagesNeg[n] = cv2.imread( join(testImagePathNeg,testFilesNeg[n]) )

    for i in range(len(testImagesNeg)):
        name= str(testFilesNeg[i])
        img = testImagesNeg[i]
        size = img.shape
        rows = size[0]
        cols = size[1]
        grayScaleImage = np.zeros((img.shape[0],img.shape[1]))
        for i in range(0,rows):
            for j in range(0,cols):
                red = img[i,j,2]
                blue = img[i,j,0]
                green = img[i,j,1]
                pixelVal = round(0.299*red + 0.587*green + 0.114 *blue)
                grayScaleImage[i,j] = pixelVal
                
        # cv2.imwrite('grayScaleImage.bmp',grayScaleImage)
        
        prewitImage, prewitGxImage, prewitGyImage, gradientAngle = prewit.prewittOperation(grayScaleImage,rows,cols)
        cv2.imwrite(name,prewitImage)
        histogram = cab.createCellHistogram(prewitImage, gradientAngle)

        HoGDescriptor = hogDesc.getHogDescriptor(histogram[0], histogram[1])
        testImageData.append(HoGDescriptor)
        testImageResult.append(0)

    # x = list(zip(imageData, imageResult))
    # random.shuffle(x)
    # testImageData, testImageResult = zip(*x)
    # Prediction using Trained Neural Network. Load already calculated values in dictionary.
    for hidden_neurons in [250,500,1000]:
        neural_network_prediction = []  #storing predicted value of the test image
        dictionary = nn.loadDictionary("valuesOfNeurons"+str(hidden_neurons))   # load model file for getting weights and bias.
	#getting all images from the list of test images and print output value of the neural network.
        print("Hidden Neurons:",hidden_neurons)
        for test_img in testImageData:
            neural_network_prediction.append(nn.predict(test_img,dictionary)) 
            if(neural_network_prediction[-1][0][0] > 0.5):
                print("Human Present in the Image:")
                print("Predicted value for Image = ",neural_network_prediction[-1][0][0])
            else:
                print("Human Not present in the image")
                print("Predicted value for Image = ",neural_network_prediction[-1][0][0])


if __name__ == '__main__':
    main()
