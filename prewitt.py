import numpy as np
import cv2
import math

def prewittOperation(grayScaleImage,rows,cols):

    #Prewit Operation Gx ,Gy and Gradient Magnitude
    prewitGx = np.array([[-1, 0 ,1],[-1, 0, 1],[-1, 0, 1]])
    prewitGxImage = np.zeros((grayScaleImage.shape[0],grayScaleImage.shape[1]))
    heightGx = (prewitGx.shape[0]-1)//2
    widthGx = (prewitGx.shape[1]-1)//2
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            tempSum = 0
            for k in range(-heightGx, heightGx + 1):
                for l in range(-widthGx , widthGx + 1):
                    tempSum = tempSum + prewitGx[heightGx+k][widthGx+l] * grayScaleImage[i+k][j+l]
            # if(tempSum < 0):
            #     tempSum = abs(tempSum)
            prewitGxImage[i][j] = tempSum / 3.0       
           
    # cv2.imwrite('PrewitGx.bmp',prewitGxImage)

    prewitGy = np.array([[1, 1 ,1],[0, 0, 0],[-1, -1, -1]])
    prewitGyImage = np.zeros((grayScaleImage.shape[0],grayScaleImage.shape[1]))
    heightGy = (prewitGy.shape[0]-1)//2
    widthGy = (prewitGy.shape[1]-1)//2

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            tempSum = 0
            for k in range(-heightGy, heightGy + 1):
                for l in range(-widthGy , widthGy + 1):
                    tempSum = tempSum + prewitGy[heightGy+k][widthGy+l] * grayScaleImage[i+k][j+l]

            # if(tempSum < 0):
            #     tempSum = abs(tempSum)  
            prewitGyImage[i][j] = tempSum / 3.0
    # cv2.imwrite('PrewitGy.bmp',prewitGyImage)

    prewittImage = np.zeros((grayScaleImage.shape[0],grayScaleImage.shape[1]))
    for i in range(1,rows - 1):
        for j in range(1,cols - 1):
            prewittImage[i][j] = ((prewitGxImage[i][j] ** 2) +  (prewitGyImage[i][j] ** 2)) ** 0.5 
            prewittImage[i][j] = prewittImage[i][j] / np.sqrt(2)

    gradientAngle = np.zeros((grayScaleImage.shape[0],grayScaleImage.shape[1]))
    sector = -1
    for i in range(1,rows - 1):
        for j in range(1,cols - 1):
            if(prewitGxImage[i][j] == 0):
                if(prewitGyImage[i][j] == 0):
                    theta = 0
                elif(prewitGyImage[i][j] < 0):
                    theta = -90
                else:
                    theta = 90
            else:
                theta = math.degrees(np.arctan(prewitGyImage[i][j] /  prewitGxImage[i][j]))
            
            if(theta < 0):
                theta = 180 + theta
            gradientAngle[i,j] = theta
            # if(theta > 0):
            #     print(theta)

            
    return prewittImage, prewitGxImage, prewitGyImage, gradientAngle
