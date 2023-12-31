import os
import cv2
import numpy as np
import bs3D
import matplotlib.pyplot as plt
import copy

def crFileName(inPath):
    '''
    
    '''
    return inPath + "\\" + os.listdir(inPath)[0]

def crRectROI(paths):
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

def getAreaCoord(paths):
    '''
    FUNCTION: get attribute and coordinate
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
    RETURN: cv2.contourArea(contours[idx]), x+int(w/2)-40, y+int(h/2)-150
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

def main2(root=".\\datat5\\"):
    labelList = bs3D.getFolderList(rootPath=root, subGroup=4)
    inputList = bs3D.getFolderList(rootPath=root, subGroup=3)
    labelFiles = []
    inputFiles = []
    fileNum = len(labelList)
    for elem in labelList:
        labelFiles.append((crFileName(elem), getAreaCoord(crFileName(elem))))
    for elem2 in inputList:
        inputFiles.append(crFileName(elem2))
    for i in range(fileNum):
        tempLabelObj = np.load(labelFiles[i][0])
        tempInputObj = np.load(inputFiles[i])["arr_0"]
        # tempLabelName = os.path.basename(labelFiles[i][0])
        # tempInputName = os.path.basename(inputFiles[i])
        tempLabelPath = os.path.abspath(labelFiles[i][0])
        # tempInputPath = os.path.abspath(inputFiles[i])
        tempIndex = tempLabelPath.split("\\")[-3]
        tempX = labelFiles[i][1][1]
        tempY = labelFiles[i][1][2]
        cropLabelImg = tempLabelObj[tempY:tempY+300,tempX:tempX+80]
        cropInputImg = tempInputObj[tempY:tempY+300,tempX:tempX+80]
        np.save(".\\cropLabelData\\"+tempIndex+"_cropLabelData",cropLabelImg)
        np.save(".\\cropInputData\\"+tempIndex+"_cropInputData",cropInputImg)

def view_fusion(num):
    root = ".\\cropLabelData2\\"
    fList = os.listdir(root)
    root2 = ".\\cropInputData2\\"
    fList2 = os.listdir(root2)
    arrlbl = np.load(root+fList[num])
    arrind = np.load(root2+fList2[num])
    mask = copy.copy(arrlbl)
    mask[arrlbl>=1]=1
    outImg = mask*arrind
    plt.imshow(outImg, cmap='gray_r')
    plt.show()

def shift_img(num, mask, inputData):
    if num >= 0 and num < 256:
        padding = np.zeros((256, num))
        transmask = np.hstack([mask[...,num:],padding])
        outimg = inputData*transmask
        plt.imshow(outimg, cmap='gray_r')
        plt.show()
    elif num >=256:
        print("ERROR, num is over 256.")
    else:
        absnum = abs(num)
        padding = np.zeros((256, absnum))
        transmask = np.hstack([padding, mask[...,:num]])
        outimg = inputData*transmask
        plt.imshow(outimg, cmap='gray_r')
        plt.show()

for i in range(num):
    tempLabelObj = np.load(lbFiles[i])
    tempInputObj = np.load(ipFiles[i])["arr_0"]
    tempIdx = lbFiles[i].split("\\")[-1][:8]
    tempX = tempCoord[i][1]
    tempY = tempCoord[i][2]
    cropLabelImg = tempLabelObj[tempY:tempY+300,tempX:tempX+80]
    cropInputImg = tempInputObj[tempY:tempY+300,tempX:tempX+80]
    # print(".\\cropLabelData\\"+tempIdx+"_%03d"%tempX+"_%03d"%tempY+"_clbD")
    # print(".\\cropInputData\\"+tempIdx+"_%03d"%tempX+"_%03d"%tempY+"_cipD")
    np.save(".\\cropLabelData\\"+tempIdx+"_%03d"%tempX+"_%03d"%tempY+"_clbD",cropLabelImg)
    np.save(".\\cropInputData\\"+tempIdx+"_%03d"%tempX+"_%03d"%tempY+"_cipD",cropInputImg)

if __name__ == "__main__":
    # fileList = bs3D.getFolderList(rootPath=".\\data5\\", subGroup=4)
    # for elem in fileList:
    #     # print(crRectROI(crFileName(elem)))
    #     print(getAreaCoord(crFileName(elem)))
    main2()
