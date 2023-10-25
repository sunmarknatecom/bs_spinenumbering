import os
import cv2
import numpy as np
import bs3D
import matplotlib.pyplot as plt
import copy
from glob import glob

def fImgView(image, mask):
    tempMask = copy.copy(mask)
    tempMask[tempMask>=1]=1
    outImg = image*tempMask
    plt.imshow(outImg, cmap='gray_r')
    plt.show()

def shiftMask(num, mask):
    y, x = np.shape(mask)
    if abs(num) >= 256:
        print("Error, shifting number out of range. Shift number must be lesser than 255")
    else:
        if num > 0:
            shiftingNum = -1 * num
            padding = np.zeros((y,num))
            temp2Mask = mask[...,:shiftingNum]
            transMask = np.hstack([padding,temp2Mask])
            return np.array(transMask, dtype=np.uint8)
        elif num == 0:
            return mask
        else:
            absNum = abs(num)
            padding = np.zeros((y,absNum))
            temp3Mask = mask[...,absNum:]
            transMask = np.hstack([temp3Mask,padding])
            return np.array(transMask, dtype=np.uint8)
        
if __name__ == "__main__":
    inDataPaths = glob(".\\inputData\\*.npz")
    lbDataPaths = glob(".\\labelData\\*.npy")
    sttNum = int(input("Start Number: "))
    endNum = int(input("End   Number: "))
    for i in range(sttNum, endNum):
        inImg = np.load(inDataPaths[i])["arr_0"]
        FNinImg = inDataPaths[i].split("\\")[-1]
        lbImg = np.load(lbDataPaths[i])
        FNlbImg = lbDataPaths[i].split("\\")[-1]
        IDX = FNinImg[:8]
        print("Loading files, ",FNinImg, "and",FNlbImg)
        fImgView(inImg, lbImg)
        continue_sign = True
        savePathLabel = ".\\modLabelData\\"
        while continue_sign == True:
            selection = input("1: realignment, q: stop \n Select option? ")
            if selection == "1":
                alignNum = int(input("Move number: neg(to left), pos(to right)"))
                tempMask = shiftMask(alignNum,lbImg)
                fImgView(inImg,tempMask)
                selection2 = input("Confirm the mask? 1: again")
                if selection2 == "1":
                    continue
                else:
                    np.save(savePathLabel+IDX+"_modLabel.npy",tempMask)
                    continue_sign = False
            elif selection == "q":
                quit()
            else:
                np.save(savePathLabel+IDX+"_modLabel.npy",lbImg)
                continue_sign = False
