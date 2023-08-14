import os
import copy
import shutil

import cv2
import numpy as np

import pydicom
import nibabel as nib
import dicom2nifti as d2n

from totalsegmentator.python_api import totalsegmentator

'''
This code is for labelling of spines on bone scan by using bone SPECT/CT data
training.
Notation rule: current array class, data type(ex: list, dict, np),
               study class(ex: NM, CT),
               data shape(ex: 3D, 2D),
               data class(ex: File, Img),
               object class(ex: Objs, Path)

Notice> note folder hierarchy.
               
Folder hierarchy: abbreviation y: year, m: month, d: day, i: index, L: level
.\data\yymm\yymmdd\yymmddii
.\L   \ L0 \ L1   \L2

To Do (pipeline)
-------------------------------------------------------------------------------
STEP1: create four folders.
        >> inputPath = getListPathL2()
        >> for elem in inputPath:
               createSubfolders(elem)
STPE2: resize raw CT data.
        >> 
        >> 
STPE3: infer the segment from resized CT data.
STEP4: transform 3D segment label data to 2D data.
STPE5: modify MVP image with data from CT and NM data.
STEP6: train the STEP4 data and STEP5 data with UNET.
STEP7: evaluate and validate the model.
-------------------------------------------------------------------------------

Rule for folder names (IDX: index, ex. 2306010101 eight digits)
-------------------------------------------------------------------------------
1. raw CT folder : variable name, multiple file
2. raw NM folder : variable name, single file
3. raw MVP folder : variable name, single file

4. resizedCT : IDX_resizedCTdcm, multiple file, respectively raw CT folder files.
5. resizedCTnii : IDX_resizedCTnii, single file
6. segData : IDX_segData, nii single file
7. inputData: IDX_inputData, npy single file
8. labelData: IDX_labelData, npy single file
'''

# data manupulation
#    folder handling
# CT resize
# convert CT dicom file to NII file
# inference of bone segmentation
#

def getListPathL2(rootPath=".\\data\\"):
    '''
    CLASS> folder handler
    NAME> getListPathL2()
    PARAMETERS> rootPath = ".\\data\\"
    RETURN>  [".\\data\\2306\\230601\\230601\\",
    '''
    root = rootPath
    pathL0 = sorted(os.listdir(root))
    retPath = []
    for elem0 in pathL0:
        pathL1 = sorted(os.listdir(os.path.join(root,elem0)))
        for elem1 in pathL1:
            pathL2 = sorted(os.listdir(os.path.join(root,elem0,elem1)))
            for elem2 in pathL2:
                pathL3 = os.path.join(root,elem0,elem1,elem2)
                retPath.append(pathL3)
    return retPath

def getListPathL2FromDict(inputDict):
    '''
    CLASS> folder handler
    NAME> getListPathL2FromDict()
    PARAMETERS> inputDict
    RETURN> [CTPath, NMPath_file, mvpPath_file]
    usually inputDict from getSubFolder()
    '''
    retList = []
    for elem in inputDict:
        for elem1 in inputDict[elem]:
            for elem2 in inputDict[elem][elem1]:
                CTPath = inputDict[elem][elem1][elem2]["CT"]
                NMPath = inputDict[elem][elem1][elem2]["NM"]+"/"+os.listdir(inputDict[elem][elem1][elem2]["NM"])[0]
                mvpPath = inputDict[elem][elem1][elem2]["MVP"]+"/"+os.listdir(inputDict[elem][elem1][elem2]["MVP"])[0]
                retList.append([CTPath, NMPath, mvpPath])
    return retList

def getDictPathL3(inputPath=".\\data\\"):
    '''
    construct the file tree dictionary
    exam) {'2306':
                  {'230601':
                            {'23060101':['ctpath','nmpath','mvppath']}
                  }
           }
    '''
    dictFolders = {}
    rootPath = inputPath
    listFolders = sorted(os.listdir(rootPath+"/")) #[(]'2306','2307']
    for elem in listFolders:
        tempRootPath = rootPath+"/"+elem+"/"
        tempPath = sorted(os.listdir(tempRootPath)) # list obj
        temp1DictFolders = {}
        # example: dictFolders {'2306':{'230601': {'23060101':['ct','nm','mvp'], '23062902', '23062903', '23062904', '23062905', '23062906', '23062907', '23062908'], '230630': ['23063001', '23063002', '23063003', '23063004', '23063005', '23063006', '23063007']}}}
        #                        elem    elem1      elem2      elem3
        # to this, {'2306':[230601, 230602, ...]}, then dictFolders[elem] = [230601, 230602, ...]
        for elem1 in tempPath:
            temp1RootPath = tempRootPath+elem1+"/"
            temp1Path = sorted(os.listdir(temp1RootPath))
            temp2DictFolders = {}
            # example temp1DictFolders = {}
            # temp1Path = "./data/230601/"
            # temp1Folders = [23060101, 23060102, ...]
            for elem2 in temp1Path:
                temp2RootPath = temp1RootPath+elem2+"/"
                temp2Path = sorted(os.listdir(temp2RootPath))
                temp3DictFolders = {}
                for elem3 in temp2Path:
                    if "CT_20" in elem3:
                        temp3DictFolders["CT"]=temp2RootPath+elem3+"/"
                    elif "TA_n" in elem3:
                        temp3DictFolders["NM"]=temp2RootPath+elem3+"/"
                    elif "MVP.P" in elem3:
                        temp3DictFolders["MVP"]=temp2RootPath+elem3+"/"
                    else:
                        break
                # example temp2DictFolders = {}
                # temp2Path = "./data/230601/23060101/"
                # temp2Folders = ['ct, 'nm', 'mvp']
                temp2DictFolders[elem2] = temp3DictFolders
            temp1DictFolders[elem1] = temp2DictFolders
        dictFolders[elem] = temp1DictFolders
    return dictFolders

def getDictPathL2(inputPath=".\\data\\"):
    '''
    CLASS> folder handler
    PARAMETERS> inputPath = ".\\data\\"
    RETURN> dictionary of folders L2
    '''
    root = inputPath
    pathL0 = sorted(os.listdir(root))
    dictFolders = {}
    for elem0 in pathL0:
        pathL1 = sorted(os.listdir(os.path.join(root,elem0)))
        tempDictL1 = {}
        for elem1 in pathL1:
            pathL2 = sorted(os.listdir(os.path.join(root,elem0,elem1)))
            tempDictL1[elem1] = pathL2
        dictFolders[elem0] = tempDictL1
    return dictFolders

def getModPath(inputList=None, subGroup=None):
    '''
    inputList is getListPathL2() result
    getModPath(subgroup=None)
    subgroup = "CT", "NM", "MVP"
    '''
    if inputList == None:
        rootPath = getListPathL2()
    else:
        rootPath = inputList
    retList = []
    filterDict = {"CT":"CT_20", "NM":"TA_n","MVP":"MVP.P","RESCT":"_resizedCTdcm","SEG":"_segData","NIFTICT":"resizedCTnii"}
    try:
        if subGroup in filterDict.keys():
            filter = filterDict[subGroup]
        for elem0 in rootPath:
            # idx = elem0.split("\\")[-1]
            subFolders = os.listdir(elem0)
            for elem1 in subFolders:
                if filter in elem1:
                    retPath = os.path.join(elem0,elem1)
                    retList.append(retPath)
        return retList
    except:
        print("subGroupError")

def createSubfolders(inputPath):
    '''
    inputPath is from getListPathL2()
    '''
    IDX = inputPath.split("\\")[-1]
    os.mkdir(os.path.join(inputPath,IDX+"_resizedCTdcm"))
    os.mkdir(os.path.join(inputPath,IDX+"_resizedCTnii"))
    os.mkdir(os.path.join(inputPath,IDX+"_segData"))
    os.mkdir(os.path.join(inputPath,IDX+"_inputData"))
    os.mkdir(os.path.join(inputPath,IDX+"_labelData"))

# def deleteSubfolders(inputPath):
#     IDX = inputPath.split("\\")[-1]
#     shutil.rmtree(os.path.join(inputPath,IDX+"_resized_segData"))
#     shutil.rmtree(os.path.join(inputPath,IDX+"_resized_inputData"))
#     shutil.rmtree(os.path.join(inputPath,IDX+"_resized_labelData"))


def showResults(init_key, last_key, len_CT, len_NM, skipped_dict):
    print("NM시작값                :",type(init_key), "   ", init_key)
    print("NM마지막 슬라이드번호   :",type(last_key), "  ",  last_key)
    print("CT 슬라이스 갯수        :",type(len_CT), "  ",  len_CT)
    print("NM 슬라이스 갯수        :",type(len_NM), "  ",  len_NM)
    print("삭제될 NM 슬라이드번호들 ",type(skipped_dict), " ",  skipped_dict)

def getObj3DCT(ctPath="./data/ct/"):
    '''
    CLASS> image processor
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
    # np.shape(CT3DImgObj)
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
    return CT3DImgObj

def getObj3DNM(nmPath="./data/nm/nm.dcm"):
    '''
    CLASS> image processor
    '''
    NMFileObj = pydicom.dcmread(nmPath)
    NM3DImgObj = NMFileObj.pixel_array
    # NM3DImgObj[imgNM3D>=300] = 300
    NM3DImgObj_transposed = np.transpose(NM3DImgObj, (1, 2, 0))
    return NM3DImgObj_transposed

def procMVPImg(ip=None, delSlices=None, needToApp=False, appNum=0, op=None):
    '''
    CLASS> image processor
    ip = input path of mvp
    delSlices = slice location list to delete
    needToApp : need to append (ex) nm size lower than CT 
    op = output path of mvp, save file format is numpy
    output path = one folder (ex: /data/result/inputData/)
    '''
    fObj = pydicom.dcmread(ip)
    fArr = fObj.pixel_array[1]
    fn = "./result/inputData/"+(os.path.split(ip))[0].split('/')[-2]
    if needToApp == False:
        saveNpy = np.delete(fArr,delSlices,axis=0)
    else:
        tempNpy = np.delete(fArr,delSlices,axis=0)
        zeroArr = np.zeros((appNum, 256))
        saveNpy = np.vstack([zeroArr,tempNpy])
    np.savez_compressed(fn, saveNpy)
    # print(np.shape(fArr))
    

def slicesToRemove(ctpath, nmpath):
    '''
    CLASS> image processor
    '''
    # slice NO
    initNum, tNumNM, tNumCT, endNumNM, keysToRemove = getTransformVar(ctPath=ctpath, nmPath=nmpath)
    # slice NO
    slices_to_remove = []
    if initNum == 1:
        prePartList = []
    else:
        prePartList = list(range(1, initNum))
    preDeleteNums = prePartList + keysToRemove
    tempLengthRemainingNM = tNumNM - len(preDeleteNums)
    needToAppend = False
    deficitNum = 0
    if tNumCT < tempLengthRemainingNM:
        diffNum = tempLengthRemainingNM - tNumCT
        tailInitNum = tNumNM-diffNum+1
        tailDeleteNums = list(range(tailInitNum,tNumNM+1))
        if preDeleteNums[-1] in tailDeleteNums:
            slices_to_remove = preDeleteNums + tailDeleteNums[1:]
            idxSTR = (np.array(slices_to_remove) - tNumNM -1)*(-1) #STR = slices_to_remove
            # print("CT < tempNM incl -1")
        else:
            slices_to_remove = preDeleteNums + tailDeleteNums
            idxSTR = (np.array(slices_to_remove) - tNumNM -1)*(-1) #STR = slices_to_remove
            # print("CT < tempNM not incl")
    elif tNumCT == tempLengthRemainingNM:
        slices_to_remove = preDeleteNums
        idxSTR = (np.array(slices_to_remove) - tNumNM -1)*(-1) #STR = slices_to_remove
        # print("CT == NM")
    else:
        slices_to_remove = preDeleteNums
        idxSTR = (np.array(slices_to_remove) - tNumNM -1)*(-1) #STR = slices_to_remove
        needToAppend = True
        deficitNum = tNumCT - tNumNM + len(idxSTR)
        # print("CT > tempNM")
    temp_len = tNumNM-len(idxSTR)
    outListSTR = list(idxSTR - 1)
    return temp_len, outListSTR, needToAppend, deficitNum, tNumCT

def rangeFilter_not_to(inputArr, n1, n2):
    '''
    CLASS> mask processor
    '''
    imgArr = np.array(inputArr, dtype=np.uint8)
    zeroArr = np.zeros_like(imgArr)
    condition = np.logical_and(imgArr>n1, imgArr<n2)
    retArr = np.where(condition, imgArr, zeroArr)
    # inputArr[inputArr>=n1 and input<=n2] = inputArr
    return retArr

def rangeFilter(inputArr, n1, n2, value):
    '''
    CLASS> mask processor
    '''
    imgArr = np.array(inputArr, dtype=np.uint8)
    zeroArr = np.zeros_like(imgArr)
    condition = np.logical_and(imgArr>=n1, imgArr<=n2)
    retArr = np.where(condition, imgArr, zeroArr)
    retArr[retArr!=0]=value
    # inputArr[inputArr>=n1 and input<=n2] = inputArr
    return retArr

def pointFilter(inputArr, n):
    '''
    CLASS> mask processor
    '''
    imgArr = np.array(inputArr, dtype=np.uint8)
    imgArr[imgArr!=n]=0
    return imgArr

def get_segRaw(inputPath):
    '''
    arrObj is (n:slice, h:height, w:width), axis=1 compress to 2D image
    '''
    fObj = nib.load(inputPath)
    tempArrObj = fObj.get_fdata()
    tempArrObj = np.transpose(tempArrObj, (2,1,0))
    arrObj = tempArrObj[::-1,::-1,::-1] #posterior view, if anterior view [::-1,::-1,::]
    # arrOjb = np.max(arrObj, axis=1)
    return arrObj

def get_segData_cervical(inputPath): # not fuse cervical spines
    '''
    arrObj is (n:slice, h:height, w:width), axis=1 compress to 2D image
    '''
    fObj = nib.load(inputPath)
    tempArrObj = fObj.get_fdata()
    tempArrObj = np.transpose(tempArrObj, (2,1,0))
    arrObj = tempArrObj[::-1,::-1,::-1] #posterior view, if anterior view [::-1,::-1,::]
    labelSpines = {"C1":41, "C2":40, "C3":39, "C4":38, "C5":37, "C6":36, "C7":35, "T1":34, "T2":33, "T3":32, "T4":31, "T5":30, "T6":29, "T7":28, "T8":27, "T9":26, "T10":25, "T11":24, "T12":23, "L1":22, "L2":21, "L3":20, "L4":19, "L5":20}
    respectiveSpines = {}
    for k, v in labelSpines.items():
        tempArr = pointFilter(arrObj, v)
        tempArr = np.max(tempArr,axis=1)
        respectiveSpines[k]=tempArr
    # arrOjb = np.max(arrObj, axis=1)
    return arrObj

def get_segData(inputPath):
    '''
    arrObj is (n:slice, h:height, w:width), axis=1 compress to 2D image
    '''
    fObj = nib.load(inputPath)
    tempArrObj = fObj.get_fdata()
    tempArrObj = np.transpose(tempArrObj, (2,1,0))
    arrObj = tempArrObj[::-1,::-1,::-1] #posterior view, if anterior view [::-1,::-1,::]
    labelSpines = {"Cervical":41, "T1":34, "T2":33, "T3":32, "T4":31, "T5":30, "T6":29, "T7":28, "T8":27, "T9":26, "T10":25, "T11":24, "T12":23, "L1":22, "L2":21, "L3":20, "L4":19, "L5":20}
    tempArrObj2 = rangeFilter(arrObj,35,41,41)
    tempArrObj2 = np.max(tempArrObj2, axis=1)
    respectiveSpines = {}
    for k, v in labelSpines.items():
        loopTempArr = pointFilter(arrObj, v)
        loopTempArr = np.max(loopTempArr,axis=1)
        respectiveSpines[k]=loopTempArr
    # arrOjb = np.max(arrObj, axis=1)
    respectiveSpines["Cervical"] = tempArrObj2
    resultArr = respectiveSpines["Cervical"]
    for k2, v2 in respectiveSpines.items():
        resultArr[v2==labelSpines[k2]]=labelSpines[k2]
    resultArr = resultArr+100
    for i in range(17):
        resultArr[resultArr==(134-i)]=i+2
    resultArr[resultArr==141]=1
    resultArr[resultArr==100]=0
    return resultArr
def getTransformVar(ctPath="./data/ct/", nmPath="./data/nm/nm.dcm"):
    '''
    CLASS>
    NAME>
    PARAMETERS>
    ctPath = ".\\data\\ct\\" dummy folder name
    nmPath = ".\\data\\nm\\" dummy folder name
    RETURN>
    keyDiffMin, list(newDictLocNM.keys())[-1], len(dictLocCT) , len(newDictLocNM), list(skippedLocNM.keys())
    DESCRIPTION>
    keyDiffMin : initial numbder of NM location
    list(newDictLocNM.keys())[-1] : last location of deleted NM slices (for rearrange between CT and NM)
    len(dictLocCT) : length of CT
    len(newDictLocNM) : length of 
    list(skippedLocNM.keys()) :
    '''
    #==============================================================================
    # CT manupulation
    # print("CT files processing")
    listCTFilePath = sorted(os.listdir(ctPath))
    listCTFileObjs = []
    for fname in listCTFilePath:
        # print("loading: {}".format(fname), end="\r")
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
    if keyDiffMin != 1:
        for key, value in dictLocNM.items():
            if found_min_key:
                newDictLocNM[key] = value
            if key == (keyDiffMin-1):
                found_min_key = True
    else:
        newDictLocNM = copy.copy(dictLocNM)
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
    # define the function to eliminate the slices of NM not to eqaulize slice location of CT
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
            # print("step2",dictLocCT[count_CT]-newDictLocNM[count_NM], diff, count_CT, count_NM)
        else:
            count_CT += 1
            count_NM += 1
            step_count +=1
            # print("step1",dictLocCT[count_CT]-newDictLocNM[count_NM], diff, count_CT, count_NM)
    for elem in keys_to_remove:
        del newDictLocNM[elem]
    # print(" init No: ", keyDiffMin, ", NM len: ", list(newDictLocNM.keys())[-1], ", CT len: ", len(dictLocCT) , ", Re NM len: ", len(newDictLocNM), list(skippedLocNM.keys()))
    return keyDiffMin, list(newDictLocNM.keys())[-1], len(dictLocCT) , len(newDictLocNM), list(skippedLocNM.keys())
    # 반환값 : (NM의 초기키값), (삭제될 NM key, value), (삭제된 NM 마지막 키값), (CT 길이), (NM길이)

def resizeCT(inputPath):
    '''
    inputPath is from getModPath(inputPath=getListPathL2, subGroup=CT)
    inputPath ".\\data\\2306\\230601\\23060101\\ctpath"
    os.path.dirname(inputPath) ".\\data\\2306\\230601\\23060101"
    '''
    IDX = inputPath.split("\\")[-2]
    fileListCT = sorted(os.listdir(inputPath))
    outputPath = os.path.join(os.path.dirname(inputPath),IDX+"_resizedCTdcm")
    # os.mkdir(outputPath)
    for fname in fileListCT:
        tempRootPath = os.path.join(inputPath,fname)
        temp_ds = pydicom.dcmread(tempRootPath)
        original_image = temp_ds.pixel_array
        resized_image = cv2.resize(original_image, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        new_ds = pydicom.dcmread(tempRootPath)
        new_ds.Rows = resized_image.shape[0]
        new_ds.Columns = resized_image.shape[1]
        new_ds.PixelSpacing = [temp_ds["PixelSpacing"].value[0]*2,temp_ds["PixelSpacing"].value[1]*2]
        new_ds.PixelData = resized_image.tobytes()
        new_ds.save_as(outputPath+"/RES_"+IDX+"_"+fname)
    print("Success ",IDX)
#labelSpines = {"C1":41, "C2":40, "C3":39, "C4":38, "C5":37, "C6":36,
#               "C7":35, "T1":34, "T2":33, "T3":32, "T4":31, "T5":30,
#               "T6":29, "T7":28, "T8":27, "T9":26, "T10":25, "T11":24,
#               "T12":23, "L1":22, "L2":21, "L3":20, "L4":19, "L5":20}

def cvt2nii(inputPath):
    '''
    CLASS> file handler
    inputPath is from getListPathL2
    inputPath ".\\data\\2306\\230601\\23060101"
    os.path.dirname ".\\data\\2306\\230601\\"

    '''
    IDX = inputPath.split("\\")[-1]
    srcPath = inputPath+"\\"+IDX+"_resizedCTdcm"
    dstPath = inputPath+'\\'+IDX+"_resizedCTnii\\"
    d2n.convert_directory(srcPath,dstPath)
    filename = os.listdir(dstPath)[0]
    os.rename(dstPath+"\\"+filename,dstPath+"\\"+IDX+"_resizedCTnii.nii.gz")
    print("Success", IDX)

def fileCollect(inputPath):
    '''
    CLASS> file handler
    inputPath is from getListPathL2
    inputPath ".\\data\\2306\\230601\\23060101"
    os.path.dirname ".\\data\\2306\\230601\\"

    '''
    IDX = inputPath.split("\\")[-1]
    srcPath = inputPath+"\\"+IDX+"_resizedCTnii\\"
    filename = os.listdir(srcPath)[0]
    shutil.copyfile(srcPath+filename,".\\data\\CTnii\\"+filename)
    print("Success ", filename)

def loadNpzFile(src):
    fObj = np.load(src)
    npArr = fObj['arr_0']
    return npArr

if __name__ == "__main__":
    # a, b, c, d, e = getTransformVar()
    # showResults(a, b, c, d, e)
    # print(a, b, c, d, e)
    errors = []
    inputPath = getDictPathL3()

    input_list = getListPathL2FromDict(inputPath)
    # def trying(input_list):
    #     try:
    #         for ctpath, nmpath,_ in input_list:
    #             A, B, C, D, E = getTransformVar(ctPath=ctpath, nmPath=nmpath)
    #     except IndexError:
    #         errors.append(ctpath)
    for i, (ctp, nmp, mvp) in enumerate(input_list):
        result_len, listIdxSTR, reqAppend, defitNum, totalNumCT = slicesToRemove(ctp, nmp)
        # print((result_len+defitNum) == totalNumCT, result_len, (os.path.split(mvp))[0].split('/')[-2])
        # print("%4d"%i, "", (result_len+defitNum) == totalNumCT, result_len, ctp)
        procMVPImg(ip=mvp, delSlices=listIdxSTR,needToApp=reqAppend,appNum=defitNum)
