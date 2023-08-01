import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

def ret_values_NM(ctPath="./data/ct/", nmPath="./data/nm/nm.dcm"):
    '''
    ctPath is folder name
    nmPath is nm file name
    notation: current array class(ex: list, dict, np), study class(ex: NM, CT),
              data shape(ex: 3D, 2D), data class(ex: File, Img),
              object class(ex: Objs, Path)
    '''
    #==============================================================================
    # CT manupulation
    print("CT files processing")
    listCTFilePath = sorted(os.listdir(ctPath))
    listCTFileObjs = []
    for fname in listCTFilePath:
        print("loading: {}".format(fname), end="\r")
        listCTFileObjs.append(pydicom.dcmread(ctPath + fname))
    print("Finished CT file processing")
    slicesCT = []
    skipCount = 0
    for f in listCTFileObjs:
        if hasattr(f, 'SliceLocation'):
            slicesCT.append(f)
        else:
            skipCount += 1
    print("skipped, no SliceLocation: {}".format(skipCount))
    slicesCT = sorted(slicesCT, key=lambda s: s.SliceLocation)
    # create 3D CT object 
    imgShapeCT = list(slicesCT[0].pixel_array.shape)
    imgShapeCT.append(len(slicesCT))
    CT3DImgObj = np.zeros(imgShapeCT)
    np.shape(imgCT3D)
    dictLocCT = {}
    for i, s in enumerate(slicesCT):
        dictLocCT[i + 1] = float(s.SliceLocation)
    for i, s in enumerate(slicesCT):
        CT2DImgObj = s.pixel_array
        CT3DImgObj[:, :, i] = CT2DImgObj
    # return data: 1. dictLocCT(slice index: location)
    #              2. listCTFile path (absolute path of CT files)
    #              3. listCTFile objects (metadata)
    #              4. CT3DImg
    #==============================================================================
    # NM manupulation
    print("NM file processing")
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
    print("NM Object slice start point searching")
    headValCT = next(iter(dictLocCT.values()))
    diffMin = float('inf')  # 초기값 설정
    keyDiffMin = None
    for key, value in dictLocNM.items():
        diff = abs(headValCT - value)
        if diff < diffMin:
            diffMin = diff
            keyDiffMin = key
    print("dictLocNM의 첫 번째 값과 차이가 가장 작은 dictLocNM value의 key:", keyDiffMin)
    # return data: 1. headValCT = 
    #==============================================================================
    # rearranged NM slices.
    # modify the NM and CT objects to same slices.
    print("Modify slices of NM and CT objects")
    newDictLocNM = {}
    found_min_key = False
    for key, value in dictLocNM.items():
        if found_min_key:
            newDictLocNM[key] = value
        if key == (keyDiffMin-1):
            found_min_key = True
    print("새로운 Dictionary:", newDictLocNM)
    dictLocCT_length = len(dictLocCT)
    newDictLocNM_length = len(newDictLocNM)
    max_length = max(dictLocCT_length, newDictLocNM_length)
    min_length = min(dictLocCT_length, newDictLocNM_length)
    if dictLocCT_length < max_length:
        for i in range(dictLocCT_length+1, max_length+1):
            dictLocCT[i] = None
    elif dictLocCT_length == max_length:
        pass
    else:
        for i in range(newDictLocNM_length+1, max_length+1):
            newDictLocNM[i] = None
    #==============================================================================
    # define the function to eliminate the slices of NM not to eqaulize slice location of CT
    count = len(dictLocNM)
    skippedLocNM = {}
    while True:
        dictLocCT, newDictLocNM, out_dict = eliminate(dictLocCT, newDictLocNM, skippedLocNM)
        if len(dictLocCT) != count:
            count = len(dictLocCT)
            print(count)
            pass
        else:
            break
    return keyDiffMin, skippedLocNM, list(newDictLocNM.keys())[-1], list(dictLocCT.keys())[-1], list(out_dict.keys())[-1]


def eliminate(dict1, dict2, outdict):
    for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
        temp_len = len(dict1)
        if value1 == None:
            print("None value")
            break
        elif abs(value1 - value2) >= 1.23:
            outdict[key2] = value2
            del dict2[key2]
            last_key = list(dict1.keys())[-1]
            del dict1[last_key]
            return dict1, dict2, outdict
            break
        else:
            pass
    return dict1, dict2, outdict

init_key, skipped_dict, last_key, lastkeys2, lastkeys3 = ret_values_NM()

print(init_key)
print(skipped_dict)
print(last_key)
