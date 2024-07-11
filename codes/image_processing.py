import os
import pydicom
import numpy as np

def load_ct_images(folder_path):
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_images = [pydicom.dcmread(file) for file in dicom_files]
    dicom_images.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    ct_positions = [float(dcm.ImagePositionPatient[2]) for dcm in dicom_images]
    return dicom_images, ct_positions

def create_ct_image_matrix(dicom_images):
    slices = len(dicom_images)
    width = dicom_images[0].Columns
    height = dicom_images[0].Rows
    image_matrix = np.zeros((slices, height, width), dtype=dicom_images[0].pixel_array.dtype)
    for i, dcm in enumerate(dicom_images):
        image_matrix[i, :, :] = dcm.pixel_array
    return image_matrix

def load_bone_scan_image(file_path):
    dicom_file = pydicom.dcmread(file_path)
    slice_thickness = dicom_file.SliceThickness
    pixel_array = dicom_file.pixel_array
    slices = pixel_array.shape[0]
    width = pixel_array.shape[2]
    height = pixel_array.shape[1]
    image_position_patient_start = dicom_file.ImagePositionPatient
    image_positions = [image_position_patient_start[2] + i * slice_thickness for i in range(slices)]
    return pixel_array, image_positions

def find_matching_slices(ct_positions, bone_scan_positions, threshold=1.23):
    ct_positions = np.array(ct_positions)
    bone_scan_positions = np.array(bone_scan_positions)
    matching_slices = []
    mismatched_slices = []
    for bone_index, bone_pos in enumerate(bone_scan_positions):
        closest_ct_index = np.argmin(np.abs(ct_positions - bone_pos))
        if abs(ct_positions[closest_ct_index] - bone_pos) <= threshold:
            matching_slices.append((bone_index, closest_ct_index))
        else:
            mismatched_slices.append((bone_index, closest_ct_index))
    return matching_slices, mismatched_slices

def get_slice_matrices(slice_indices, ct_image_matrix, bone_scan_image_matrix):
    ct_slices = []
    bone_slices = []
    for bone_index, ct_index in slice_indices:
        bone_slices.append(bone_scan_image_matrix[bone_index])
        ct_slices.append(ct_image_matrix[ct_index])
    return ct_slices, bone_slices

ct_folder_path = '.\\temp\\CT_n751\\'
bone_scan_file_path = '.\\temp\\BS_n758\\' + os.listdir('.\\temp\\BS_n758\\')[0]

# CT DICOM 파일 불러오기 및 정렬
ct_images, ct_positions = load_ct_images(ct_folder_path)
ct_image_matrix = create_ct_image_matrix(ct_images)

# Bone Scan DICOM 파일 불러오기
bone_scan_image_matrix, bone_scan_positions = load_bone_scan_image(bone_scan_file_path)

# match 및 mismatch된 Bone Scan slice 번호 찾기
matching_slices, mismatched_slices = find_matching_slices(ct_positions, bone_scan_positions)
print("Mismatched Bone Scan slices:", mismatched_slices)
print("Matching Bone Scan slices:", matching_slices)

# match된 slice들의 CT와 Bone Scan matrix 반환
ct_matching_slices, bone_matching_slices = get_slice_matrices(matching_slices, ct_image_matrix, bone_scan_image_matrix)

# 결과 출력
print(f"Number of matching slices: {len(matching_slices)}")
print(f"Number of mismatched slices: {len(mismatched_slices)}")

# Matching CT와 Bone Scan 배열 출력
print("Matching CT slices matrix:")
for i, ct_slice in enumerate(ct_matching_slices):
    print(f"CT slice {i}:")
    print(ct_slice)

print("Matching Bone scan slices matrix:")
for i, bone_slice in enumerate(bone_matching_slices):
    print(f"Bone scan slice {i}:")
    print(bone_slice)
