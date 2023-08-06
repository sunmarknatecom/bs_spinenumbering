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


