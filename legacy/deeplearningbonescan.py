import os
import copy
import cv2
import numpy as np
import pydicom
import nibabel as nib
from glob import glob
from totalsegmentator.python_api import totalsegmentator


class MedicalImageProcessor:
    def __init__(self, rootPath=".\\data\\"):
        self.rootPath = rootPath

    def _get_filtered_folder_list(self, subGroup):
        """
        Helper function to get a list of directories filtered by a specific pattern.

        Parameters:
        subGroup (str): Pattern to filter directories.

        Returns:
        list: Sorted list of directory names that match the pattern.
        """
        filtered_dirs = []
        for dirpath, _, _ in os.walk(self.rootPath):
            if subGroup in dirpath:
                filtered_dirs.append(dirpath)
        return sorted(filtered_dirs)

    def _create_directories(self, base_path, dir_names):
        """
        Helper function to create multiple directories.

        Parameters:
        base_path (str): Base path where directories will be created.
        dir_names (list): List of directory names to create.
        """
        for dir_name in dir_names:
            os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)

    def getFolderList(self, subGroup=0):
        """
        Retrieve a list of directories matching a specific subgroup pattern.

        Parameters:
        subGroup (int): Index of the subgroup pattern to match.

        Returns:
        list: Sorted list of directory names that match the subgroup pattern.
        """
        subGroupList = ["_CT_", ".MVP.Planar", ".TA_", "inputData", "labelData", "resizedCTdcm", "resizedCTnii", "segData"]
        return self._get_filtered_folder_list(subGroupList[subGroup])

    def getIdxFolderList(self):
        """
        Retrieve a list of index folder paths within the specified root directory.

        Returns:
        list: List of index folder paths.
        """
        temp_list = self.getFolderList()
        return sorted(set(os.path.dirname(elem) for elem in temp_list))

    def getIdxFolderListFromDict(self, inputDict):
        """
        Extract a list of index folder paths from a nested dictionary structure.

        Parameters:
        inputDict (dict): Input nested dictionary containing folder paths.

        Returns:
        list: List of index folder paths.
        """
        retList = []
        for elem in inputDict.values():
            for elem1 in elem.values():
                for elem2 in elem1.values():
                    CTPath = elem2['CT']
                    NMPath = elem2['NM'] + '\\' + os.listdir(elem2['NM'])[0]
                    mvpPath = elem2['MVP'] + '\\' + os.listdir(elem2['MVP'])[0]
                    retList.append([CTPath, NMPath, mvpPath])
        return retList

    def getDictPathL3(self):
        """
        Create a nested dictionary structure representing folder paths up to the third level.

        Returns:
        dict: Nested dictionary containing folder paths up to the third level.
        """
        dictFolders = {}
        listFolders = sorted(os.listdir(self.rootPath))
        for elem in listFolders:
            tempRootPath = os.path.join(self.rootPath, elem)
            temp1DictFolders = {}
            for elem1 in sorted(os.listdir(tempRootPath)):
                temp1RootPath = os.path.join(tempRootPath, elem1)
                temp2DictFolders = {}
                for elem2 in sorted(os.listdir(temp1RootPath)):
                    temp2RootPath = os.path.join(temp1RootPath, elem2)
                    temp3DictFolders = {}
                    for elem3 in sorted(os.listdir(temp2RootPath)):
                        if "CT_20" in elem3:
                            temp3DictFolders["CT"] = os.path.join(temp2RootPath, elem3)
                        elif "TA_n" in elem3:
                            temp3DictFolders["NM"] = os.path.join(temp2RootPath, elem3)
                        elif "MVP.P" in elem3:
                            temp3DictFolders["MVP"] = os.path.join(temp2RootPath, elem3)
                    temp2DictFolders[elem2] = temp3DictFolders
                temp1DictFolders[elem1] = temp2DictFolders
            dictFolders[elem] = temp1DictFolders
        return dictFolders

    def getDictPathL2(self):
        """
        Create a nested dictionary structure representing folder paths up to the second level.

        Returns:
        dict: Nested dictionary containing folder paths up to the second level.
        """
        dictFolders = {}
        for elem0 in sorted(os.listdir(self.rootPath)):
            pathL1 = sorted(os.listdir(os.path.join(self.rootPath, elem0)))
            tempDictL1 = {elem1: sorted(os.listdir(os.path.join(self.rootPath, elem0, elem1))) for elem1 in pathL1}
            dictFolders[elem0] = tempDictL1
        return dictFolders

    def getModPath(self, subGroup):
        """
        Retrieve a list of paths for a specific subgroup within the given input list.

        Parameters:
        subGroup (str): Subgroup key to filter paths.

        Returns:
        list: List of paths matching the specified subgroup.
        """
        filterDict = {"CT": "CT_20", "NM": "TA_n", "MVP": "MVP.P", "RESCT": "_resizedCTdcm", "SEG": "_segData", "NIFTICT": "resizedCTnii"}
        filter_key = filterDict.get(subGroup, "")
        return self._get_filtered_folder_list(filter_key)

    def createSubFolders(self, inputPath):
        """
        Create subfolders for resized CT, NIfTI, and segmentation data.

        Parameters:
        inputPath (str): Root directory to create subfolders in.
        """
        IDX = os.path.basename(inputPath)
        subfolder_names = [f"{IDX}_{suffix}" for suffix in ["resizedCTdcm", "resizedCTnii", "segData", "inputData", "labelData"]]
        self._create_directories(inputPath, subfolder_names)

    def showResults(self, init_key, last_key, len_CT, len_NM, skipped_dict):
        """
        Display results of the data processing steps.

        Parameters:
        init_key (int): Initial key value.
        last_key (int): Last key value.
        len_CT (int): Length of the CT data.
        len_NM (int): Length of the NM data.
        skipped_dict (dict): Dictionary of skipped slices.
        """
        print("NM 시작값                :", type(init_key), "   ", init_key)
        print("NM 마지막 슬라이드번호   :", type(last_key), "  ", last_key)
        print("CT 슬라이스 갯수        :", type(len_CT), "  ", len_CT)
        print("NM 슬라이스 갯수        :", type(len_NM), "  ", len_NM)
        print("삭제될 NM 슬라이드번호들 ", type(skipped_dict), " ", skipped_dict)

    def getObj3DCT(self, ctPath):
        """
        Process CT DICOM files and create a 3D CT object.

        Parameters:
        ctPath (str): Path to the directory containing CT DICOM files.

        Returns:
        np.ndarray: 3D array representing the CT images.
        """
        print("CT files processing")
        listCTFileObjs = [pydicom.dcmread(os.path.join(ctPath, fname)) for fname in sorted(os.listdir(ctPath))]
        pxArrays = np.array([s.pixel_array for s in listCTFileObjs])
        return pxArrays

    def getObj3DNM(self, nmPath="./data/nm"):
        """
        Process NM DICOM files and create a 3D NM object.

        Parameters:
        nmPath (str): Path to the directory containing NM DICOM files.

        Returns:
        np.ndarray: 3D array representing the NM images.
        """
        print("NM file processing")
        nmRef = pydicom.dcmread(nmPath)
        pxArrays = np.array(nmRef.pixel_array)
        return pxArrays

    def procMVPImg(self, ip="./data", delSlices=[], needToApp=False):
        """
        Process MVP images by deleting specified slices.

        Parameters:
        ip (str): Path to the directory containing MVP images.
        delSlices (list): List of slice indices to delete.
        needToApp (bool): Whether to append additional processing.

        Returns:
        None
        """
        tempNpy = np.load(ip)
        tempNpy = np.delete(tempNpy, list(delSlices), axis=2)
        saveNpy = copy.deepcopy(tempNpy)
        fn = os.path.join(ip, "..", "..", "..", "labelData")
        if needToApp:
            exNpy = np.load(fn + "\\" + os.path.splitext(os.path.basename(ip))[0] + ".npy")
            saveNpy = np.stack([exNpy, tempNpy])
        os.makedirs(fn, exist_ok=True)
        print("saving npy file")
        np.save(fn + "\\" + os.path.splitext(os.path.basename(ip))[0], saveNpy)

    def create3DObjFile(self, inputPath="./"):
        """
        Create 3D object files for CT and NM data, and process MVP images.

        Parameters:
        inputPath (str): List containing paths to CT, NM, and MVP data.

        Returns:
        None
        """
        try:
            CTPath, NMPath, MVPPath = inputPath
        except:
            print("Failed to unpack inputPath")
            return 1
        CTObj = self.getObj3DCT(CTPath)
        NMObj = self.getObj3DNM(NMPath)
        CTShape = CTObj.shape
        NMShape = NMObj.shape
        skipVal = 0
        dictSkipped = {}
        delSlice = 0
        for i in range(NMShape[2]):
            key = i + 1
            if key not in dictLocCT.values():
                dictSkipped[key] = skipVal + 1
                skipVal += 1
                delSlice += 1
        self.procMVPImg(ip=MVPPath, delSlices=dictSkipped.values(), needToApp=False)
        self.showResults(init_key=1, last_key=NMShape[2], len_CT=CTShape[2], len_NM=NMShape[2], skipped_dict=dictSkipped)

    def createNiiResizedCTFile(self):
        """
        Create resized CT files in NIfTI format.

        Returns:
        None
        """
        CTFiles = self.getModPath(subGroup="CT")
        for CT in CTFiles:
            outdcm = CT.split("\\")[-2] + "_resizedCTdcm"
            outnft = CT.split("\\")[-2] + "_resizedCTnii"
            outputDirDcm = os.path.join(os.path.dirname(CT), outdcm)
            outputDirNii = os.path.join(os.path.dirname(CT), outnft)
            self._create_directories(os.path.dirname(CT), [outdcm, outnft])
            self.resizeCT(CTPath=CT, saveDcm=outputDirDcm, saveNii=outputDirNii)

    def resizeCT(self, CTPath, saveDcm, saveNii, target_shape=(256, 256)):
        """
        Resizes CT images and saves them in both DICOM and NIfTI formats.

        Parameters:
        - CTPath (str): Path to the directory containing CT DICOM files.
        - saveDcm (str): Path to the directory where resized DICOM files will be saved.
        - saveNii (str): Path to the directory where resized NIfTI file will be saved.
        - target_shape (tuple): Target shape for resizing (height, width).

        Returns:
        None
        """
        dicom_files = sorted(glob(os.path.join(CTPath, "*.dcm")))
        resized_images = []

        for dicom_file in dicom_files:
            ds = pydicom.dcmread(dicom_file)
            image = ds.pixel_array
            resized_image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized_image)

            # Save resized image back to DICOM format
            ds.PixelData = resized_image.tobytes()
            ds.Rows, ds.Columns = resized_image.shape
            ds.save_as(os.path.join(saveDcm, os.path.basename(dicom_file)))

        # Convert the list of resized images to a 3D numpy array
        resized_images_3d = np.stack(resized_images, axis=-1)

        # Save the 3D numpy array as a NIfTI file
        nii_image = nib.Nifti1Image(resized_images_3d, affine=np.eye(4))
        nib.save(nii_image, os.path.join(saveNii, 'resized_image.nii'))

    def createSegmentLabelData(self):
        """
        Create segmentation label data using TotalSegmentator.

        Returns:
        None
        """
        inputCT = self.getModPath(subGroup="NIFTICT")
        outputCT = self.getModPath(subGroup="SEG")
        for i in range(len(inputCT)):
            totalsegmentator(input_path=inputCT[i], output_path=outputCT[i])

    def prepareSegmentLabelData(self, inputSeg):
        """
        Prepare segmentation label data for further processing or analysis.

        Parameters:
        inputSeg (list): List of segmentation data paths.

        Returns:
        None
        """
        inputSeg = self.getModPath(subGroup="SEG")
        for elem in inputSeg:
            subFolders = sorted(os.listdir(elem))
            for i, seg in enumerate(subFolders):
                fObj = nib.load(os.path.join(elem, seg))
                fArr = fObj.get_fdata()
                seg_out_path = os.path.join(elem, "..", "..", "..", "_inputData", f"{seg}_input.npy")
                os.makedirs(os.path.dirname(seg_out_path), exist_ok=True)
                np.save(seg_out_path, fArr)
                if i == 0:
                    seg_out_path = os.path.join(elem, "..", "..", "..", "_labelData", f"{seg}_label.npy")
                    os.makedirs(os.path.dirname(seg_out_path), exist_ok=True)
                    np.save(seg_out_path, fArr)


if __name__ == "__main__":
    processor = MedicalImageProcessor()
    processor.create3DObjFile(inputPath=processor.getIdxFolderListFromDict(inputDict=processor.getDictPathL3()))
    processor.createNiiResizedCTFile()
    processor.createSegmentLabelData()
    processor.prepareSegmentLabelData(inputSeg=processor.getModPath(subGroup="SEG"))
