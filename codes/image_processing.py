import os
import pydicom
import numpy as np

def load_ct_images(folder_path):
    """Loads CT DICOM files from a folder and sorts them by ImagePositionPatient[2].

    Args:
        folder_path (str): Path to the folder containing CT DICOM files.

    Returns:
        tuple: (list of pydicom objects, list of ImagePositionPatient[2] values)
    """
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_images = [pydicom.dcmread(file) for file in dicom_files]
    dicom_images.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    ct_positions = [float(dcm.ImagePositionPatient[2]) for dcm in dicom_images]
    return dicom_images, ct_positions

def create_ct_image_matrix(dicom_images):
    """Creates a 3D numpy array from a list of CT DICOM images.

    Args:
        dicom_images (list): List of pydicom objects.

    Returns:
        np.ndarray: 3D numpy array of shape (slices, height, width).
    """
    slices = len(dicom_images)
    width = dicom_images[0].Columns
    height = dicom_images[0].Rows
    image_matrix = np.zeros((slices, height, width), dtype=dicom_images[0].pixel_array.dtype)
    for i, dcm in enumerate(dicom_images):
        image_matrix[i, :, :] = dcm.pixel_array
    return image_matrix

def load_bone_scan_image(file_path):
    """Loads a bone scan DICOM file and extracts the 3D image array and positions.

    Args:
        file_path (str): Path to the bone scan DICOM file.

    Returns:
        tuple: (3D numpy array, list of ImagePositionPatient[2] values)
    """
    dicom_file = pydicom.dcmread(file_path)
    slice_thickness = dicom_file.SliceThickness
    pixel_array = dicom_file.pixel_array
    slices = pixel_array.shape[0]
    image_position_patient_start = dicom_file.ImagePositionPatient
    image_positions = [image_position_patient_start[2] + i * slice_thickness for i in range(slices)]
    return pixel_array, image_positions

def find_matching_slices(ct_positions, bone_scan_positions, threshold=1.23):
    """Finds matching and mismatched slices based on ImagePositionPatient[2] values.

    Args:
        ct_positions (list): List of ImagePositionPatient[2] values for CT images.
        bone_scan_positions (list): List of ImagePositionPatient[2] values for bone scan images.
        threshold (float): Maximum allowable difference for a match.

    Returns:
        tuple: (list of matching slice indices, list of mismatched slice indices)
    """
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
    """Retrieves the image matrices for the given slice indices.

    Args:
        slice_indices (list): List of tuples containing bone scan and CT slice indices.
        ct_image_matrix (np.ndarray): 3D numpy array of CT images.
        bone_scan_image_matrix (np.ndarray): 3D numpy array of bone scan images.

    Returns:
        tuple: (list of 2D numpy arrays for CT slices, list of 2D numpy arrays for bone scan slices)
    """
    ct_slices = []
    bone_slices = []
    for bone_index, ct_index in slice_indices:
        bone_slices.append(bone_scan_image_matrix[bone_index])
        ct_slices.append(ct_image_matrix[ct_index])
    return ct_slices, bone_slices

def main():
    """Main function to process CT and bone scan DICOM files."""
    ct_folder_path = '.\\temp\\CT_n751\\'
    bone_scan_file_path = '.\\temp\\BS_n758\\' + os.listdir('.\\temp\\BS_n758\\')[0]

    # Load and sort CT DICOM files
    ct_images, ct_positions = load_ct_images(ct_folder_path)
    ct_image_matrix = create_ct_image_matrix(ct_images)

    # Load bone scan DICOM file
    bone_scan_image_matrix, bone_scan_positions = load_bone_scan_image(bone_scan_file_path)

    # Find matching and mismatched slices
    matching_slices, mismatched_slices = find_matching_slices(ct_positions, bone_scan_positions)
    print("Mismatched Bone Scan slices:", mismatched_slices)
    print("Matching Bone Scan slices:", matching_slices)

    # Retrieve matrices for matching slices
    ct_matching_slices, bone_matching_slices = get_slice_matrices(matching_slices, ct_image_matrix, bone_scan_image_matrix)

    # Output results
    print(f"Number of matching slices: {len(matching_slices)}")
    print(f"Number of mismatched slices: {len(mismatched_slices)}")

    print("Matching CT slices matrix:")
    for i, ct_slice in enumerate(ct_matching_slices):
        print(f"CT slice {i}:")
        print(ct_slice)

    print("Matching Bone scan slices matrix:")
    for i, bone_slice in enumerate(bone_matching_slices):
        print(f"Bone scan slice {i}:")
        print(bone_slice)

if __name__ == '__main__':
    main()
