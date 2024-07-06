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
    height, width = np.shape(tempMask)
    rHeight = 12
    rWidth = 12 * width / height
    plt.figure(figsize=(rWidth, rHeight))
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
    print("Total object range is from 1 to 670.")
    sttNum = int(input("What is start Number?  = "))
    endNum = int(input("What is end number?    = "))
    for i in range(sttNum-1, endNum):
        inImg = np.load(inDataPaths[i])["arr_0"]
        FNinImg = inDataPaths[i].split("\\")[-1]
        lbImg = np.load(lbDataPaths[i])
        FNlbImg = lbDataPaths[i].split("\\")[-1]
        IDX = FNinImg[:8]
        print(i+1, ",  Loading index, ", IDX)
        fImgView(inImg, lbImg)
        continue_sign = True
        savePathLabel = ".\\modLabelData\\"
        while continue_sign == True:
            selection = input("   \"1\": Realignment, \"z\": Exit, Any key: Next \n Select option? ")
            if selection == "1":
                alignNum = int(input("        Move number: neg(to left), pos(to right) ? "))
                tempMask = shiftMask(alignNum,lbImg)
                fImgView(inImg,tempMask)
                selection2 = input("        Confirm the mask? \"1\": again, Any key: save and next ? ")
                if selection2 == "1":
                    continue
                elif selection == "z":
                    print("GOOD BYE!!!")
                    quit()
                else:
                    np.save(savePathLabel+IDX+"_modLabel.npy",tempMask)
                    print("Complete save", IDX+"_modLabel.npy\n----------------------------------------------")
                    continue_sign = False
            elif selection == "z":
                print("GOOD BYE!!!")
                quit()
            else:
                np.save(savePathLabel+IDX+"_modLabel.npy",lbImg)
                print("Complete save", IDX+"_modLabel.npy\n----------------------------------------------")
                continue_sign = False
    print("End of processing!!!")
