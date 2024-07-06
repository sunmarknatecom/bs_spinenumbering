import pydicom
import numpy as np
import os
import cv2
import dicom2nifti as d2n
import shutil

dictFolders = {}

rootPath = "./data/"

listFolders = os.listdir(rootPath)

for elem in listFolders:
    dictFolders[elem] = os.listdir(rootPath + elem)
# example: dictFolders {'230629': ['23062901', '23062902', '23062903', '23062904', '23062905', '23062906', '23062907', '23062908'], '230630': ['23063001', '23063002', '23063003', '23063004', '23063005', '23063006', '23063007']}

for elem in dictFolders:
    tempDictFolders = {}
    for elem2 in dictFolders[elem]:
        tempDictFolders2 = {}
        tempPath = rootPath+elem+"/"+elem2+"/"
        tempFolders = os.listdir(tempPath)
        tempFolders2 = {}
        for elem3 in tempFolders:
            if "CT_20" in elem3:
                tempFolders2["CT"]=tempPath+elem3
            elif "TA_n" in elem3:
                tempFolders2["NM"]=tempPath+elem3
            elif "MVP.P" in elem3:
                tempFolders2["MVP"]=tempPath+elem3
            elif "RES" in elem3:
                tempFolders2["RES"]=tempPath+elem3
            elif "NIFTI_CT" in elem3:
                tempFolders2["NIFTI"]=tempPath+elem3
            else:
                pass
        tempDictFolders[elem2]=tempFolders2
    dictFolders[elem] = tempDictFolders

foldersCT = []
foldersNM = []
foldersMVP = []
foldersRES = []
foldersNIFTI = []

for elem in dictFolders:
    for elem2 in dictFolders[elem]:
        for elem3 in dictFolders[elem][elem2]:
            if elem3 == "CT":
                foldersCT.append(dictFolders[elem][elem2][elem3])

for elem in dictFolders:
    for elem2 in dictFolders[elem]:
        for elem3 in dictFolders[elem][elem2]:
            if elem3 == "NM":
                foldersNM.append(dictFolders[elem][elem2][elem3])

for elem in dictFolders:
    for elem2 in dictFolders[elem]:
        for elem3 in dictFolders[elem][elem2]:
            if elem3 == "MVP":
                foldersMVP.append(dictFolders[elem][elem2][elem3])

for elem in dictFolders:
    for elem2 in dictFolders[elem]:
        for elem3 in dictFolders[elem][elem2]:
            if elem3 == "RES":
                foldersRES.append(dictFolders[elem][elem2][elem3])

for elem in dictFolders:
    for elem2 in dictFolders[elem]:
        for elem3 in dictFolders[elem][elem2]:
            if elem3 == "NIFTI":
                foldersNIFTI.append(dictFolders[elem][elem2][elem3])


# create nifti folders

def createNiftiFolder(inputPath):
    fileListCT = sorted(os.listdir(inputPath))
    outputPath = os.path.dirname(inputPath)+'/'+os.path.basename(os.path.dirname(inputPath))+"_NIFTI_CT"
    os.mkdir(outputPath)

def resizeCT(inputPath):
    fileListCT = sorted(os.listdir(inputPath))
    outputPath = os.path.dirname(inputPath)+'/'+os.path.basename(os.path.dirname(inputPath))+"_RESIZED_CT"
    os.mkdir(outputPath)
    objName = os.path.basename(os.path.dirname(inputPath))
    for fname in fileListCT:
        temp_ds = pydicom.dcmread(inputPath+"/"+fname)
        original_image = temp_ds.pixel_array
        resized_image = cv2.resize(original_image, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        new_ds = pydicom.dcmread(inputPath+"/"+fname)
        new_ds.Rows = resized_image.shape[0]
        new_ds.Columns = resized_image.shape[1]
        new_ds.PixelData = resized_image.tobytes()
        new_ds.save_as(outputPath+"/"+'RES_'+objName+"_"+fname)

for elem in foldersCT:
    resizeCT(elem)

for elem in foldersCT:
    createNiftiFolder(elem)

foldersRES

def convert2nifti(inputPath):
    outputPath = os.path.dirname(inputPath)+'/'+os.path.basename(os.path.dirname(inputPath))+"_NIFTI_CT"
    d2n.convert_directory(inputPath,outputPath)

def rename_nifti(inputPath):
    filename = os.listdir(inputPath)
    srcFileName = inputPath+"/"+filename[0]
    trgFileName = "./nifti/" + os.path.basename(inputPath) + ".nii.gz"
    shutil.copy(srcFileName, trgFileName)
