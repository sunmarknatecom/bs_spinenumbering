import os
import cv2
import numpy as np
import bs3D

def crtFN(inPath):
    '''
    
    '''
    return inPath + "\\" + os.listdir(inPath)[0]

def createRectROI(paths):
    '''
    FUNCTION: createRectROI
    INPUTDATA TYPE: path
    PARAMS: rootPath=".\\data\\", subGroup=4
    subGroup=
    (0 --> raw CT data 512X512 sized (multiple files)
     1 --> MVP image (one file)
     2 --> raw bone SPECT data (one file)
     3 --> inputData, modified MVP image (one file, 2 images)
     4 --> labelData, 2D segmented label image (one file)
     5 --> resizedCTdcm, 256X256 sized (multiple file, .dcm)
     6 --> resizedCTnii, 256X256 sized (one file, .nii.gz)
     7 --> segData, infered 3D label data (one file, .nii)
    RETURN: cv2.contourArea(contours[idx]), x, y, w, h
    '''
    tempObj = np.load(paths)
    contours, h_ = cv2.findContours(tempObj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp_arr = []
    for elem in contours:
        temp_va = cv2.contourArea(elem)
        temp_arr.append(temp_va)
    temp_np = np.array(temp_arr)
    idx = np.argmax(temp_np)
    x, y, w, h = cv2.boundingRect(contours[idx])
    # print(cv2.contourArea(contours[idx]), idx, x, y, w, h)
    return cv2.contourArea(contours[idx]), x, y, w, h

def createRegRectROI(paths):
    '''
    FUNCTION: createRectROI
    INPUTDATA TYPE: path
    PARAMS: rootPath=".\\data\\", subGroup=4
    subGroup=
    (0 --> raw CT data 512X512 sized (multiple files)
     1 --> MVP image (one file)
     2 --> raw bone SPECT data (one file)
     3 --> inputData, modified MVP image (one file, 2 images)
     4 --> labelData, 2D segmented label image (one file)
     5 --> resizedCTdcm, 256X256 sized (multiple file, .dcm)
     6 --> resizedCTnii, 256X256 sized (one file, .nii.gz)
     7 --> segData, infered 3D label data (one file, .nii)
    RETURN: cv2.contourArea(contours[idx]), x, y, w, h
    '''
    tempObj = np.load(paths)
    contours, h_ = cv2.findContours(tempObj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp_arr = []
    for elem in contours:
        temp_va = cv2.contourArea(elem)
        temp_arr.append(temp_va) 
    temp_np = np.array(temp_arr)
    idx = np.argmax(temp_np)
    x, y, w, h = cv2.boundingRect(contours[idx])
    # print(cv2.contourArea(contours[idx]), idx, x, y, w, h)
    return cv2.contourArea(contours[idx]), x+int(w/2)-40, y+int(h/2)-150

if __name__ == "__main__":
    fileList = bs3D.getFolderList(rootPath=".\\data5\\", subGroup=4)
    for elem in fileList:
        # print(createRectROI(crtFN(elem)))
        print(createRegRectROI(crtFN(elem)))
