import numpy as np
import cv2
import math

def createCellHistogram(gradientVal, gradientAngle):
    # Calculated GradientMagnitude and Angle will be passed to this method.
    # We will make create Cell histogram where each cell has 8*8 pixels.
    # Based on the gradient angle value we will assign a bin to it.
    cell = 8
    rows = gradientVal.shape[0]
    cols = gradientVal.shape[1]
    cellR = round(rows/cell)
    cellC = round(cols/cell)
    cellHist = np.zeros((cellR,cellC,9))
    for i in range (1,cellR-1):
        for j in range (1,cellC-1):
            for rows in range (i*8,i*8+8):
                for cols in range (j*8,j*8+8):
                    angle = gradientAngle[rows,cols]
                    mag = gradientVal[rows,cols]
                    if(angle == 0 or angle == 180):
                        cellHist[i,j,0] += mag
                        continue
                    if(angle > 0 and angle < 20):
                        cellHist[i,j,0] += ((20-angle)/20)*mag
                        cellHist[i,j,1] += ((angle-0)/20)*mag
                        continue
                    if(angle == 20):
                        cellHist[i,j,1] += mag
                        continue
                    if(angle > 20 and angle < 40):
                        cellHist[i,j,1] += ((40-angle)/20)*mag
                        cellHist[i,j,2] += ((angle-20)/20)*mag
                        continue
                    if(angle == 40):
                        cellHist[i,j,2] += mag
                        continue
                    if(angle > 40 and angle < 60):
                        cellHist[i,j,2] += ((60-angle)/20)*mag
                        cellHist[i,j,3] += ((angle-40)/20)*mag
                        continue
                    if(angle == 60):
                        cellHist[i,j,3] += mag
                        continue
                    if(angle > 60 and angle < 80):
                        cellHist[i,j,3] += ((80-angle)/20)*mag
                        cellHist[i,j,4] += ((angle-60)/20)*mag
                        continue
                    if(angle == 80):
                        cellHist[i,j,4] += mag
                        continue
                    if(angle > 80 and angle < 100):
                        cellHist[i,j,4] += ((100-angle)/20)*mag
                        cellHist[i,j,5] += ((angle-80)/20)*mag
                        continue
                    if(angle == 100):
                        cellHist[i,j,5] += mag
                        continue
                    if(angle > 100 and angle < 120):
                        cellHist[i,j,5] += ((120-angle)/20)*mag
                        cellHist[i,j,6] += ((angle-100)/20)*mag
                        continue
                    if(angle == 120):
                        cellHist[i,j,6] += mag
                        continue
                    if(angle > 120 and angle < 140):
                        cellHist[i,j,6] += ((140-angle)/20)*mag
                        cellHist[i,j,7] += ((angle-120)/20)*mag
                        continue
                    if(angle == 140):
                        cellHist[i,j,7] += mag
                        continue
                    if(angle > 140 and angle < 160):
                        cellHist[i,j,7] += ((160-angle)/20)*mag
                        cellHist[i,j,8] += ((angle-140)/20)*mag
                        continue
                    if(angle == 160):
                        cellHist[i,j,8] += mag
                        continue
                    if(angle > 160):
                        cellHist[i,j,8] += ((180-angle)/20)*mag
                        cellHist[i,j,0] += ((angle-160)/20)*mag
                        continue
    cellHistSqrd = np.square(cellHist)
    return [cellHist, cellHistSqrd]