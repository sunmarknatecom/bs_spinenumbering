import os
import numpy as np
import pydicom

def get_ct_3d_object(ct_path="./data/ct/"):
    """
    Process CT files and return a 3D numpy array representation.
    
    Parameters:
    ct_path (str): The path to the CT files.

    Returns:
    numpy.ndarray: A 3D numpy array representing the CT image.
    """
    print("Processing CT files")
    ct_file_list = sorted(os.listdir(ct_path))
    ct_file_objects = []
    for file_name in ct_file_list:
        print(f"Loading: {file_name}", end="\r")
        ct_file_objects.append(pydicom.dcmread(ct_path + file_name))
    print("Finished processing CT files")
    ct_slices = []
    skip_count = 0
    for file in ct_file_objects:
        if hasattr(file, 'SliceLocation'):
            ct_slices.append(file)
        else:
            skip_count += 1
    print(f"Skipped files without SliceLocation: {skip_count}")
    ct_slices = sorted(ct_slices, key=lambda s: s.SliceLocation)
    ct_shape = list(ct_slices[0].pixel_array.shape)
    ct_shape.append(len(ct_slices))
    ct_3d_object = np.zeros(ct_shape)
    ct_location_dict = {}
    for i, slice in enumerate(ct_slices):
        ct_location_dict[i + 1] = float(slice.SliceLocation)
    for i, slice in enumerate(ct_slices):
        ct_2d_image = slice.pixel_array
        ct_3d_object[:, :, i] = ct_2d_image
    return ct_3d_object

def get_nm_3d_object(nm_path="./data/nm/nm.dcm"):
    """
    Process an NM file and return a 3D numpy array representation.
    
    Parameters:
    nm_path (str): The path to the NM file.

    Returns:
    numpy.ndarray: A 3D numpy array representing the NM image.
    """
    nm_file_object = pydicom.dcmread(nm_path)
    nm_3d_object = nm_file_object.pixel_array
    nm_3d_object_transposed = np.transpose(nm_3d_object, (1, 2, 0))
    return nm_3d_object_transposed

def process_mvp_image(input_path=None, delete_slices=None, need_to_append=False, append_num=0, output_path=None):
    """
    Process an MVP image by deleting specific slices and optionally appending zero slices.
    
    Parameters:
    input_path (str): The path to the MVP file.
    delete_slices (list): The list of slice indices to delete.
    need_to_append (bool): Whether to append zero slices.
    append_num (int): The number of zero slices to append.
    output_path (str): The path to save the processed image.

    Returns:
    None
    """
    file_object = pydicom.dcmread(input_path)
    file_array = file_object.pixel_array[1]
    file_name = ".\\result\\inputData\\" + (os.path.split(input_path))[0].split('\\')[-2]
    if not need_to_append:
        save_array = np.delete(file_array, delete_slices, axis=0)
    else:
        temp_array = np.delete(file_array, delete_slices, axis=0)
        zero_array = np.zeros((append_num, 256))
        save_array = np.vstack([zero_array, temp_array])
    np.savez_compressed(file_name, save_array)

def slices_to_remove(ct_path, nm_path):
    """
    Determine the slices to remove from the NM image based on the CT image.

    Parameters:
    ct_path (str): The path to the CT files.
    nm_path (str): The path to the NM file.

    Returns:
    list: A list of slice indices to remove.
    """
    init_num, total_nm_slices, total_ct_slices, end_num_nm, keys_to_remove = get_transform_variables(ct_path=ct_path, nm_path=nm_path)
    slices_to_remove = []
    if init_num == 1:
        pre_part_list = []
    else:
        pre_part_list = list(range(1, init_num))
    pre_delete_nums = pre_part_list + keys_to_remove
    remaining_nm_slices = total_nm_slices - len(pre_delete_nums)
    need_to_append = False
    deficit_num = 0
    if total_ct_slices < remaining_nm_slices:
        diff_num = remaining_nm_slices - total_ct_slices
        tail_init_num = total_nm_slices - diff_num + 1
        tail_delete_nums = list(range(tail_init_num, total_nm_slices + 1))
        pre_delete_nums += tail_delete_nums
    return pre_delete_nums
