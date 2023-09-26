dictLocCT = {}
newDictLocNM = {}

for i in range(715):
    dictLocCT[i] = 2.5*i+364.11

for j in range(758):
    newDictLocNM[j+41] = 2.46*j+363.74

len_CT = len(dictLocCT)
#len_NM = len(newDictLocNM)
len_NM = 758
count_CT = 1
count_NM = 41

keys_CT = list(dictLocCT.keys())
keys_NM = list(newDictLocNM.keys())

keys_to_remove = []

step_count = 0

while (count_CT <= len_CT) and (count_NM <= len_NM):
    diff = dictLocCT[count_CT]-newDictLocNM[count_NM]
    if diff >= 1.23:
        keys_to_remove.append(count_NM)
        count_NM +=1
        step_count +=1
        print("step2",dictLocCT[count_CT]-newDictLocNM[count_NM], diff, count_CT, count_NM)
    else:
        count_CT += 1
        count_NM += 1
        step_count +=1
        print("step1",dictLocCT[count_CT]-newDictLocNM[count_NM], diff, count_CT, count_NM)


import os
import numpy as np
import pydicom
import bs3d
import copy

errors=[]
inputPath = bs3d.getSubFolders()
input_list = bs3d.inputList(inputPath)
# for ctpath, nmpath in input_list:
#     bs3d.ret_values_NM(ctPath=ctpath, nmPath=nmpath)


ctPath = input_list[0][0]
nmPath = input_list[0][-1]
listCTFilePath = sorted(os.listdir(ctPath))
listCTFileObjs = []
for fname in listCTFilePath:
    print("loading: {}".format(fname), end="\r")
    listCTFileObjs.append(pydicom.dcmread(ctPath + fname))

slicesCT = []
skipCount = 0
for f in listCTFileObjs:
    if hasattr(f, 'SliceLocation'):
        slicesCT.append(f)
    else:
        skipCount += 1

slicesCT = sorted(slicesCT, key=lambda s: s.SliceLocation)

imgShapeCT = list(slicesCT[0].pixel_array.shape)
imgShapeCT.append(len(slicesCT))

dictLocCT = {}
for i, s in enumerate(slicesCT):
    dictLocCT[i + 1] = float(s.SliceLocation)

NMFileObj = pydicom.dcmread(nmPath)
locationNM = float(NMFileObj["ImagePositionPatient"].value[2]) # location
NM3DImgObj = NMFileObj.pixel_array
# NM3DImgObj[imgNM3D>=300] = 300
NM3DImgObj_transposed = np.transpose(NM3DImgObj, (1, 2, 0))
dictLocNM = {}
nmSliceThickness = float(NMFileObj.SliceThickness)
lenNM = np.shape(NM3DImgObj_transposed)[2]

for i in range(lenNM):
    dictLocNM[i + 1] = float(locationNM + i * nmSliceThickness)


# print("NM Object slice start point searching")
headValCT = next(iter(dictLocCT.values()))
diffMin = float('inf')  # 초기값 설정
keyDiffMin = None
for key, value in dictLocNM.items():
    diff = abs(headValCT - value)
    if diff < diffMin:
        diffMin = diff
        keyDiffMin = key


newDictLocNM = {}
found_min_key = False
if keyDiffMin != 1:
    for key, value in dictLocNM.items():
        if found_min_key:
            newDictLocNM[key] = value
        if key == (keyDiffMin-1):
            found_min_key = True
else:
    newDictLocNM = copy.copy(dictLocNM)

skippedLocNM = {}
len_CT = len(dictLocCT)
len_NM = list(newDictLocNM.keys())[-1]
count_CT = 1
count_NM = list(newDictLocNM.keys())[0]
keys_to_remove = []
step_count = 0

while (count_CT <= len_CT) and (count_NM <= len_NM):
    diff = dictLocCT[count_CT]-newDictLocNM[count_NM]
    if diff >= 1.23:
        keys_to_remove.append(count_NM)
        skippedLocNM[count_NM]=newDictLocNM[count_NM]
        count_NM +=1
        step_count +=1
        print("step2",dictLocCT[count_CT]-newDictLocNM[count_NM], diff, count_CT, count_NM)
    else:
        count_CT += 1
        count_NM += 1
        step_count +=1
        print("step1",dictLocCT[count_CT]-newDictLocNM[count_NM], diff, count_CT, count_NM)
for elem in keys_to_remove:
    del newDictLocNM[elem]
print(keyDiffMin, list(newDictLocNM.keys())[-1], len(dictLocCT) , len(newDictLocNM), list(skippedLocNM.keys()))
# 반환값 : (NM의 초기키값), (삭제될 NM key, value), (삭제된 NM 마지막 키값), (CT 길이), (NM길이)


errors = []
inputPath = bs3d.getSubFolders()
input_list = bs3d.inputList(inputPath)

initNum, tNumNM, tNumCT, endNumNM, keysToRemove = bs3d.ret_values_NM(ctPath=ctpath, nmPath=nmpath)

slices_to_remove = []

prePartList = list(range(1, initNum))

preDeleteNums = prePartList + keysToRemove

tempSlicesToRemove = tNumNM - len(preDeleteNums)

diffNum = tNumCT - tempSlicesToRemove

tailDeleteNums = list(range(tNumNM-diffNum+1,tNumNM+1))

slices_to_remove = preDeleteNums + tailDeleteNums

idxSTR = (np.array(slices_to_remove) - tNumNM)*(-1)

# Img = mvp의 posterior image

outImg = np.delete(Img, idxSTR, axis=0)


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
    RETURN: cv2.contourArea(contours[idx]), y+int(h/2)-128
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
    if x+int(w/2)-128 <= 0:
        ret_x = 0
    else:
        ret_x = x+int(w/2)-128
    return cv2.contourArea(contours[idx]), ret_x, y+int(h/2)-128

lbFiles = sorted(glob(".\\labelData\\*.npy"))
ipFiles = sorted(glob(".\\inputData\\*.npz"))

tempCoord = []

num = len(lbFiles)

for i in range(num):
    tempCoord.append(getAreaCoord(lbFiles[i]))

for i in range(num):
    tempLabelObj = np.load(lbFiles[i])
    tempInputObj = np.load(ipFiles[i])["arr_0"]
    tempInputObj = np.array(tempInputObj, dtype=np.uint16)
    tempIdx = lbFiles[i].split("\\")[-1][:8]
    tempX = tempCoord[i][1]
    tempY = tempCoord[i][2]
    cropLabelImg = tempLabelObj[tempY:tempY+256,...]
    cropInputImg = tempInputObj[tempY:tempY+256,...]
    # print(".\\cropLabelData\\"+tempIdx+"_%03d"%tempX+"_%03d"%tempY+"_clbD")
    # print(".\\cropInputData\\"+tempIdx+"_%03d"%tempX+"_%03d"%tempY+"_cipD")
    np.save(".\\cropLabelData2\\"+tempIdx+"_%03d"%tempY+"_clbD",cropLabelImg)
    np.save(".\\cropInputData2\\"+tempIdx+"_%03d"%tempY+"_cipD",cropInputImg)
