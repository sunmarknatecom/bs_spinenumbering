import os
import pydicom
import numpy as np
import copy

# ctPath="./data/ct/"
# nmPath="./data/nm/nm.dcm"
'''
ctPath is folder name
nmPath is nm file name
notation: current array class(ex: list, dict, np), study class(ex: NM, CT),
            data shape(ex: 3D, 2D), data class(ex: File, Img),
            object class(ex: Objs, Path)
'''
#==============================================================================
# CT manupulation
# print("CT files processing")
ctPath="D:/BONESPECT/data/2306/230602/23060201/23060201_ANON77424_CT_2023-06-02_154452_Other_Bone_n715__00000/"
nmPath="D:/BONESPECT/data/2306/230602/23060201/23060201_ANON77424_NM_2023-06-02_154128_Whole.Body.Bone.SCH_WBS6.FFS.MDP.OSAC.SCH1.TA_n758__00000/2.16.840.1.114362.1.12122241.23963427765.647525879.1008.9760.dcm"
# ctPath = "D:/BONESPECT/data/2306/230601/23060101/23060101_ANON58452_CT_2023-06-01_162545_Other_Bone_n668__00000/"
# nmPath = "D:/BONESPECT/data/2306/230601/23060101/23060101_ANON58452_NM_2023-06-01_162407_Whole.Body.Bone.SCH_WBS6.FFS.MDP.OSAC.SCH1.TA_n758__00000/2.16.840.1.114362.1.12122241.23963427765.647526826.820.2644.dcm"
listCTFilePath = sorted(os.listdir(ctPath))
listCTFileObjs = []
for fname in listCTFilePath:
    print("loading: {}".format(fname), end="\r")
    listCTFileObjs.append(pydicom.dcmread(ctPath + fname))

# print("Finished CT file processing")
slicesCT = []
skipCount = 0
for f in listCTFileObjs:
    if hasattr(f, 'SliceLocation'):
        slicesCT.append(f)
    else:
        skipCount += 1

# print("skipped, no SliceLocation: {}".format(skipCount))
slicesCT = sorted(slicesCT, key=lambda s: s.SliceLocation)
# create 3D CT object 
imgShapeCT = list(slicesCT[0].pixel_array.shape)
imgShapeCT.append(len(slicesCT))
# CT3DImgObj = np.zeros(imgShapeCT)
# np.shape(CT3DImgObj)
dictLocCT = {}
for i, s in enumerate(slicesCT):
    dictLocCT[i + 1] = float(s.SliceLocation)

# for i, s in enumerate(slicesCT):
#     CT2DImgObj = s.pixel_array
#     CT3DImgObj[:, :, i] = CT2DImgObj
# return data: 1. dictLocCT(slice index: location)
#              2. listCTFile path (absolute path of CT files)
#              3. listCTFile objects (metadata)
#              4. CT3DImg
#==============================================================================
# NM manupulation
# print("NM file processing")
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

# return data: 1. NMFileObj
#              2. NM3DImgObj (generally, h X w X slices)
#              3. NM3DImgObj_transposed (generally, slices X h X w)
#              4. dictLocNM (index: sliceLoction)
#==============================================================================
# search CT-NM start point
# dictionary keys를 list로 변환시 순서가 안 바뀐다는 가정
# print("NM Object slice start point searching")
headValCT = next(iter(dictLocCT.values()))
diffMin = float('inf')  # 초기값 설정
keyDiffMin = None
for key, value in dictLocNM.items():
    diff = abs(headValCT - value)
    if diff < diffMin:
        diffMin = diff
        keyDiffMin = key

# print("dictLocNM의 첫 번째 값과 차이가 가장 작은 dictLocNM value의 key:", keyDiffMin)
# return data: 1. headValCT = first value of CT slices
#              2. diffMin = minimum value of diff between CT and NM
#              3. keyDiffMin = key of minimum value of diff between CT and NM
#==============================================================================
# rearranged NM slices.
# modify the NM and CT objects to same slices.
# print("Modify slices of NM and CT objects")
newDictLocNM = {}
found_min_key = False
for key, value in dictLocNM.items():
    if found_min_key:
        newDictLocNM[key] = value
    if key == (keyDiffMin-1):
        found_min_key = True

#작동원리: key:1, value: 255이면, keyDiffMin가 77일때,
# found_min_key가 false이므로 그냥 돌다가 76에서 found_min_key로 바뀌면 입력시작
# print("새로운 Dictionary:", newDictLocNM)
# dictLocCT_length = len(dictLocCT)
# newDictLocNM_length = len(newDictLocNM)
# max_length = max(dictLocCT_length, newDictLocNM_length)
# min_length = min(dictLocCT_length, newDictLocNM_length)
# if dictLocCT_length < max_length:
#     for i in range(dictLocCT_length+1, max_length+1):
#         dictLocCT[i] = None
# elif dictLocCT_length == max_length:
#     pass
# else:
#     for i in range(newDictLocNM_length+1, max_length+1):
#         newDictLocNM[i] = None
#==============================================================================
if found_min_key == False:
    newDictLocNM = copy.copy(dictLocNM)

#==============================================================================
#==============================================================================
# define the function to eliminate the slices of NM not to eqaulize slice location of CT
skippedLocNM = {}
keys_to_remove = []
prev_key_CT = 1
prev_key_NM = None

for key_CT, value_CT in dictLocCT.items():
    if prev_key_NM is not None and key_CT - prev_key_CT > 1:
        prev_key_NM = None
        # print("1st if", key_CT, prev_key_NM, key_NM)
    if prev_key_NM is None:
        for key_NM, value_NM in newDictLocNM.items():
            if abs(value_CT-value_NM) <= 1.23:
                prev_key_NM = key_NM
                # print("2nd if", key_CT, prev_key_NM, key_NM)
                break
    if prev_key_NM is not None:
        while prev_key_NM in newDictLocNM:
            value_NM = newDictLocNM[prev_key_NM]
            if abs(value_CT-value_NM) > 1.23:
                skippedLocNM[prev_key_NM] = newDictLocNM[prev_key_NM]
                keys_to_remove.append(prev_key_NM)
                prev_key_NM += 1
                # print("3rd if", key_CT, prev_key_NM, key_NM, "diff: ", value_CT-value_NM)
            else:
                prev_key_NM += 1
                # print("4th if ", key_CT, prev_key_NM, key_NM)
                break
    prev_key_CT = key_CT

for elem in keys_to_remove:
    del newDictLocNM[elem]

print(keyDiffMin, list(newDictLocNM.keys())[-1], len(dictLocCT) , len(newDictLocNM), list(skippedLocNM.keys()))
# 반환값 : (NM의 초기키값), (삭제될 NM key, value), (삭제된 NM 마지막 키값), (CT 길이), (NM길이)

dictLocCT = {}
newDictLocNM = {}

for i in range(715):
    dictLocCT[i] = 2.5*i+364.11

for j in range(718):
    newDictLocNM[j] = 2.46*i+363.74

skippedLocNM = {}
keys_to_remove = []
keys_to_remove = []
prev_key_CT = None
prev_key_NM = None
for key_CT, value_CT in dictLocCT.items():
    if prev_key_NM is not None and key_CT - prev_key_CT > 1:
        prev_key_NM = None
    if prev_key_NM is None:
        for key_NM, value_NM in newDictLocNM.items():
            if abs(value_CT-value_NM) <= 1.23:
                prev_key_NM = key_NM
                break
    if prev_key_NM is not None:
        while prev_key_NM in newDictLocNM:
            value_NM = newDictLocNM[prev_key_NM]
            if abs(value_CT-value_NM) > 1.23:
                skippedLocNM[prev_key_NM] = newDictLocNM[prev_key_NM]
                keys_to_remove.append(prev_key_NM)
                prev_key_NM += 1
            else:
                prev_key_NM += 1
                break
    prev_key_CT = key_CT


