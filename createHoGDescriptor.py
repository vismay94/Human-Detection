import numpy as np
import cv2
import math

def getHogDescriptor(cellHist, cellHistSqrd):
    # Created CellHistogram will be passed here. We will make block of 4*4 cell.
    # Block will iterate over call and It will include 50% of already included cellHistogram. 
    rows = cellHist.shape[0]
    cols = cellHist.shape[1]
    descriptor = np.array([])
    for i in range(0,rows-1):
        for j in range(0,cols-1):
            block = np.array([])
            temp = np.array([])
            block = np.append(block,cellHist[i,j])
            block = np.append(block,cellHist[i,j+1])
            block = np.append(block,cellHist[i+1,j])
            block = np.append(block,cellHist[i+1,j+1])
            temp = np.append(temp,cellHistSqrd[i,j])
            temp = np.append(temp,cellHistSqrd[i,j+1])
            temp = np.append(temp,cellHistSqrd[i+1,j])
            temp = np.append(temp,cellHistSqrd[i+1,j+1])
            temp = np.sum(temp)
            if(temp>0):
                    #normalize the block descriptor
                norm = np.sqrt(temp)
                block = (1/norm)*block
            descriptor = np.append(descriptor, block)
    return descriptor