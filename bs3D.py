import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

def ret_values_NM(ctPath="./data/ct/", nmPath="./data/nm/nm.dcm"):
    '''
    ctPath is folder name
    nmPath is nm file name
    '''
    #==============================================================================
    # CT manupulation
    print("CT files processing")
    fileListCT = sorted(os.listdir(ctPath))
    fileObjsCT = []
    for fname in fileListCT:
        print("loading: {}".format(fname))
        fileObjsCT.append(pydicom.dcmread(ctPath + fname))
    slicesCT = []
    skipCount = 0
    for f in fileObjsCT:
        if hasattr(f, 'SliceLocation'):
            slicesCT.append(f)
        else:
            skipCount += 1
    print("skipped, no SliceLocation: {}".format(skipCount))
    slicesCT = sorted(slicesCT, key=lambda s: s.SliceLocation)
    # create 3D CT object 
    imgShapeCT = list(slicesCT[0].pixel_array.shape)
    imgShapeCT.append(len(slicesCT))
    imgCT3D = np.zeros(imgShapeCT)
    np.shape(imgCT3D)
    dictLocCT = {}
    for i, s in enumerate(slicesCT):
        dictLocCT[i + 1] = float(s.SliceLocation)
    for i, s in enumerate(slicesCT):
        imgCT2D = s.pixel_array
        imgCT3D[:, :, i] = imgCT2D
    #==============================================================================
    # NM manupulation
    print("NM file processing")
    fileObjNM = pydicom.dcmread(nmPath)
    locationNM = float(fileObjNM["ImagePositionPatient"].value[2]) # location
    imgNM3D = fileObjNM.pixel_array
    #imgNM3D[imgNM3D>=300] = 300
    imgNM3DTransposed = np.transpose(imgNM3D, (1, 2, 0))
    dictLocNM = {}
    nmSliceThickness = float(fileObjNM.SliceThickness)
    lenNM = np.shape(imgNM3DTransposed)[2]
    for i in range(lenNM):
        dictLocNM[i + 1] = float(locationNM + i * nmSliceThickness)
    #==============================================================================
    # search CT-NM start point
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
