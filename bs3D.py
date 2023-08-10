import os
import pydicom
import numpy as np
import copy
import nibabel as nib
import cv2
import dicom2nifti as d2n
import shutil

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

def CT3DObj(ctPath="./data/ct/"):
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

def NM3DObj(nmPath="./data/nm/nm.dcm"):
    NMFileObj = pydicom.dcmread(nmPath)
    NM3DImgObj = NMFileObj.pixel_array
    # NM3DImgObj[imgNM3D>=300] = 300
    NM3DImgObj_transposed = np.transpose(NM3DImgObj, (1, 2, 0))
    return NM3DImgObj_transposed

def getFolders(option=False):
    '''
    construct the file tree dictionary
    exam) {'2306':
                  {'230601':
                            {'23060101':['ctpath','nmpath','mvppath']}
                  }
           }
    '''
    dictFolders = {}
    rootPath = "./data"
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
                    elif "RES" in elem3:
                        temp3DictFolders["RES"]=temp2RootPath+elem3+"/"
                    else:
                        break
                # example temp2DictFolders = {}
                # temp2Path = "./data/230601/23060101/"
                # temp2Folders = ['ct, 'nm', 'mvp']
                temp2DictFolders[elem2] = temp3DictFolders
            temp1DictFolders[elem1] = temp2DictFolders
        dictFolders[elem] = temp1DictFolders
    foldersCT = []
    foldersNM = []
    foldersMVP = []
    foldersRES = []
    for elem in dictFolders:
        for elem2 in dictFolders[elem]:
            for elem3 in dictFolders[elem][elem2]:
                for elem4 in dictFolders[elem][elem2][elem3]:
                    if elem4 == "CT":
                        foldersCT.append(dictFolders[elem][elem2][elem3][elem4])
    for elem in dictFolders:
        for elem2 in dictFolders[elem]:
            for elem3 in dictFolders[elem][elem2]:
                for elem4 in dictFolders[elem][elem2][elem3]:
                    if elem4 == "NM":
                        foldersNM.append(dictFolders[elem][elem2][elem3][elem4])
    for elem in dictFolders:
        for elem2 in dictFolders[elem]:
            for elem3 in dictFolders[elem][elem2]:
                for elem4 in dictFolders[elem][elem2][elem3]:
                    if elem4 == "MVP":
                        foldersMVP.append(dictFolders[elem][elem2][elem3][elem4])
    for elem in dictFolders:
        for elem2 in dictFolders[elem]:
            for elem3 in dictFolders[elem][elem2]:
                for elem4 in dictFolders[elem][elem2][elem3]:
                    if elem4 == "RES":
                        foldersRES.append(dictFolders[elem][elem2][elem3][elem4])
    if option=="CT":
        return foldersCT
    elif option=="NM":
        return foldersNM
    elif option=="MVP":
        return foldersMVP
    else:
        return dictFolders

def getSubFolders():
    '''
    construct the file tree dictionary
    exam) {'2306':
                  {'230601':
                            {'23060101':['ctpath','nmpath','mvppath']}
                  }
           }
    '''
    dictFolders = {}
    rootPath = "./data"
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
                    elif "RES" in elem3:
                        temp3DictFolders["RES"]=temp2RootPath+elem3+"/"
                    else:
                        break
                # example temp2DictFolders = {}
                # temp2Path = "./data/230601/23060101/"
                # temp2Folders = ['ct, 'nm', 'mvp']
                temp2DictFolders[elem2] = temp3DictFolders
            temp1DictFolders[elem1] = temp2DictFolders
        dictFolders[elem] = temp1DictFolders
    return dictFolders

def getRootPath():
    '''
    construct the file tree dictionary
    exam) {'2306':
                  {'230601':
                            {'23060101':['ctpath','nmpath','mvppath']}
                  }
           }
    '''
    dictFolders = {}
    rootPath = "./data"
    listFolders = sorted(os.listdir(rootPath+"/")) #[(]'2306','2307']
    for elem in listFolders:
        tempRootPath = rootPath+"/"+elem+"/"
        tempPath = sorted(os.listdir(tempRootPath)) # list obj
        temp1DictFolders = {}
        # example: dictFolders {'2306':{'230601': {'23060101':['ct','nm','mvp'], '23062902', '23062903', '23062904', '23062905', '23062906', '23062907', '23062908'], '230630': ['23063001', '23063002', '23063003', '23063004', '23063005', '23063006', '23063007']}}}
        #                        elem    elem1      elem2      elem3
        # to this, {'2306':[230601, 230602, ...]}, then dictFolders[elem] = [230601, 230602, ...]
        for elem1 in tempPath:
            temp1RootPath = rootPath+"/"+elem+"/"+elem1+"/"
            temp1Path = sorted(os.listdir(temp1RootPath))
            temp2DictFolders = {}
            # example temp1DictFolders = {}
            # temp1Path = "./data/230601/"
            # temp1Folders = [23060101, 23060102, ...]
            for elem2 in temp1Path:
                temp2Path = rootPath+"/"+elem+"/"+elem1+"/"+elem2+"/"
                temp2Folders = sorted(os.listdir(temp2Path))
                temp2DictFolders[elem2] = temp2Folders
                # example temp2DictFolders = {}
                # temp2Path = "./data/230601/23060101/"
                # temp2Folders = ['ct, 'nm', 'mvp']
            temp1DictFolders[elem1] = temp2DictFolders
        dictFolders[elem] = temp1DictFolders
    return dictFolders

def inputList(a):
    '''
    a is a getsubfolders
    '''
    input_list = []
    for elem in a:
        for elem1 in a[elem]:
            for elem2 in a[elem][elem1]:
                NMpath = a[elem][elem1][elem2]["NM"]+"/"+os.listdir(a[elem][elem1][elem2]["NM"])[0]
                mvppath = a[elem][elem1][elem2]["MVP"]+"/"+os.listdir(a[elem][elem1][elem2]["MVP"])[0]
                input_list.append([a[elem][elem1][elem2]["CT"], NMpath, mvppath])
    return input_list

# for elem0 in pathL0:
#     pathL1 = sorted(os.listdir(os.path.join(root,elem0)))
#     tempDictL1 = {}
#     for elem1 in pathL1:
#         pathL2 = sorted(os.listdir(os.path.join(root,elem0,elem1)))
#         tempDictL2 = {}
#         for elem2 in pathL2:
#             pathL3 = sorted(os.listdir(os.path.join(root,elem0,elem1,elem2)))
#             tempDictL2[elem2] = pathL3
#         tempDictL1[elem1] = tempDictL2
#     dictFolders[elem0] = tempDictL1

def getDictPath():
    root = ".\\data\\"
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

def getListPath():
    '''
    return to ".\\data\\2306\\230601\\230601\\"
    '''
    root = ".\\data\\"
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

def getModPath(inputList=None, subGroup=None):
    '''
    inputList is getListPath() result
    getModPath(subgroup=None)
    subgroup = "CT", "NM", "MVP"
    '''
    if inputList == None:
        rootPath = getListPath()
    else:
        rootPath = inputList
    retList = []
    filterDict = {"CT":"CT_20", "NM":"TA_n","MVP":"MVP.P","RESCT":"RESCT","SEG":"SEG","NIFTICT":"NIFTICT"}
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
    
def getTuplePath():
    pass


                # for elem3 in temp2Path:
                #     if "CT_20" in elem3:
                #         temp3DictFolders["CT"]=temp2RootPath+elem3+"/"
                #     elif "TA_n" in elem3:
                #         temp3DictFolders["NM"]=temp2RootPath+elem3+"/"
                #     elif "MVP.P" in elem3:
                #         temp3DictFolders["MVP"]=temp2RootPath+elem3+"/"
                #     elif "RES" in elem3:
                #         temp3DictFolders["RES"]=temp2RootPath+elem3+"/"
                #     else:
                #         break

def createSubfolders(inputPath):
    '''
    inputPath is from getListPath()
    '''
    IDX = inputPath.split("\\")[-1]
    os.mkdir(os.path.join(inputPath,IDX+"_resized_CT_dicom"))
    os.mkdir(os.path.join(inputPath,IDX+"_resized_CT_nii"))
    os.mkdir(os.path.join(inputPath,IDX+"_segData"))
    os.mkdir(os.path.join(inputPath,IDX+"_inputData"))
    os.mkdir(os.path.join(inputPath,IDX+"_labelData"))

# def deleteSubfolders(inputPath):
#     IDX = inputPath.split("\\")[-1]
#     shutil.rmtree(os.path.join(inputPath,IDX+"_resized_segData"))
#     shutil.rmtree(os.path.join(inputPath,IDX+"_resized_inputData"))
#     shutil.rmtree(os.path.join(inputPath,IDX+"_resized_labelData"))


def results(init_key, last_key, len_CT, len_NM, skipped_dict):
    print("NM시작값                :",type(init_key), "   ", init_key)
    print("NM마지막 슬라이드번호   :",type(last_key), "  ",  last_key)
    print("CT 슬라이스 갯수        :",type(len_CT), "  ",  len_CT)
    print("NM 슬라이스 갯수        :",type(len_NM), "  ",  len_NM)
    print("삭제될 NM 슬라이드번호들 ",type(skipped_dict), " ",  skipped_dict)

def mvpImgProc(ip=None, delSlices=None, needToApp=False, appNum=0, op=None):
    '''
    ip = input path of mvp
    delSlices = slice location list to delete
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
    # slice NO
    initNum, tNumNM, tNumCT, endNumNM, keysToRemove = ret_values_NM(ctPath=ctpath, nmPath=nmpath)
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

def segProcess(inputPath):
    fObj = nib.load(inputPath)
    arrObj = fObj.get_fdata()
    pass

def rangeFilter_not_to(inputArr, n1, n2):
    imgArr = np.array(inputArr, dtype=np.uint8)
    zeroArr = np.zeros_like(imgArr)
    condition = np.logical_and(imgArr>n1, imgArr<n2)
    retArr = np.where(condition, imgArr, zeroArr)
    # inputArr[inputArr>=n1 and input<=n2] = inputArr
    return retArr

def rangeFilter(inputArr, n1, n2, value):
    imgArr = np.array(inputArr, dtype=np.uint8)
    zeroArr = np.zeros_like(imgArr)
    condition = np.logical_and(imgArr>=n1, imgArr<=n2)
    retArr = np.where(condition, imgArr, zeroArr)
    retArr[retArr!=0]=value
    # inputArr[inputArr>=n1 and input<=n2] = inputArr
    return retArr

def pointFilter(inputArr, n):
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

def resizeCT(inputPath):
    '''
    inputPath is from getModPath(inputPath=getListPath, subGroup=CT)
    inputPath ".\\data\\2306\\230601\\23060101\\ctpath"
    os.path.dirname(inputPath) ".\\data\\2306\\230601\\23060101"
    '''
    IDX = inputPath.split("\\")[-2]
    fileListCT = sorted(os.listdir(inputPath))
    outputPath = os.path.join(os.path.dirname(inputPath),IDX+"_resized_CT_dicom")
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
#labelSpines = {"C1":41, "C2":40, "C3":39, "C4":38, "C5":37, "C6":36, "C7":35, "T1":34, "T2":33, "T3":32, "T4":31, "T5":30, "T6":29, "T7":28, "T8":27, "T9":26, "T10":25, "T11":24, "T12":23, "L1":22, "L2":21, "L3":20, "L4":19, "L5":20}

def convert2nifti(inputPath):
    '''
    inputPath is from getListPath
    inputPath ".\\data\\2306\\230601\\23060101"
    os.path.dirname ".\\data\\2306\\230601\\"

    '''
    IDX = inputPath.split("\\")[-1]
    tempInputPath = inputPath+"\\"+IDX+"_resized_CT_dicom"
    outputPath = inputPath+'\\'+IDX+"_resized_CT_nii\\"
    d2n.convert_directory(tempInputPath,outputPath)
    filename = os.listdir(outputPath)[0]
    os.rename(outputPath+"\\"+filename,outputPath+"\\"+IDX+"_resized_CT_nii.nii.gz")
    print("Success", IDX)

def fileCollect(inputPath):
    '''
    inputPath is from getListPath
    inputPath ".\\data\\2306\\230601\\23060101"
    os.path.dirname ".\\data\\2306\\230601\\"

    '''
    IDX = inputPath.split("\\")[-1]
    srcPath = inputPath+"\\"+IDX+"_resized_CT_nii\\"
    filename = os.listdir(srcPath)[0]
    shutil.copyfile(srcPath+filename,".\\data\\CTnii\\"+filename)
    print("Success ", filename)

if __name__ == "__main__":
    # a, b, c, d, e = ret_values_NM()
    # results(a, b, c, d, e)
    # print(a, b, c, d, e)
    errors = []
    inputPath = getSubFolders()

    input_list = inputList(inputPath)
    # def trying(input_list):
    #     try:
    #         for ctpath, nmpath,_ in input_list:
    #             A, B, C, D, E = ret_values_NM(ctPath=ctpath, nmPath=nmpath)
    #     except IndexError:
    #         errors.append(ctpath)
    for i, (ctp, nmp, mvp) in enumerate(input_list):
        result_len, listIdxSTR, reqAppend, defitNum, totalNumCT = slicesToRemove(ctp, nmp)
        # print((result_len+defitNum) == totalNumCT, result_len, (os.path.split(mvp))[0].split('/')[-2])
        # print("%4d"%i, "", (result_len+defitNum) == totalNumCT, result_len, ctp)
        mvpImgProc(ip=mvp, delSlices=listIdxSTR,needToApp=reqAppend,appNum=defitNum)
