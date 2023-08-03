import os
import pydicom
import numpy as np

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
    # define the function to eliminate the slices of NM not to eqaulize slice location of CT
    skippedLocNM = {}
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
    for elem in keys_to_remove:
        del newDictLocNM[elem]
    print(keyDiffMin, list(newDictLocNM.keys())[-1], len(dictLocCT) , len(newDictLocNM), list(skippedLocNM.keys()))
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

def legacyFunction(ctPath="./data/ct/", nmPath="./data/nm/nm.dcm"):
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
    # return data: 1. headValCT = first value of CT slices
    #              2. diffMin = minimum value of diff between CT and NM
    #              3. keyDiffMin = key of minimum value of diff between CT and NM
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

    for elem in keys_to_remove:
        del newDictLocNM[elem]
    
    return keyDiffMin, list(newDictLocNM.keys())[-1], len(dictLocCT) , len(newDictLocNM), list(skippedLocNM.keys())
    # 반환값 : (NM의 초기키값), (삭제될 NM key, value), (삭제된 NM 마지막 키값), (CT 길이), (NM길이)

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
                input_list.append([a[elem][elem1][elem2]["CT"], NMpath])
    return input_list


def results(init_key, last_key, len_CT, len_NM, skipped_dict):
    print("NM시작값                :",type(init_key), "   ", init_key)
    print("NM마지막 슬라이드번호   :",type(last_key), "  ",  last_key)
    print("CT 슬라이스 갯수        :",type(len_CT), "  ",  len_CT)
    print("NM 슬라이스 갯수        :",type(len_NM), "  ",  len_NM)
    print("삭제될 NM 슬라이드번호들 ",type(skipped_dict), " ",  skipped_dict)

if __name__ == "__main__":
    # a, b, c, d, e = ret_values_NM()
    # results(a, b, c, d, e)
    # print(a, b, c, d, e)
    errors = []
    def trying(input_list):
        try:
            for ctpath, nmpath in input_list:
                A, B, C, D, E = ret_values_NM(ctPath=ctpath, nmPath=nmpath)
                print(A, B, C, D, E)
        except IndexError:
            errors.append(ctpath)
