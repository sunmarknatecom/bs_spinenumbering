import file_handling as fh
import image_processing as ip

def main():
    """
    Main function demonstrating the usage of file handling and image processing functions.
    """
    # Example usage of file handling functions
    root_path = ".\\data\\"
    sub_group = 0
    folder_list = fh.get_folder_list(root_path, sub_group)
    print(f"Folder List: {folder_list}")

    index_folder_list = fh.get_index_folder_list(root_path)
    print(f"Index Folder List: {index_folder_list}")

    dict_path_l3 = fh.get_dict_path_l3(root_path)
    print(f"Dict Path L3: {dict_path_l3}")

    dict_path_l2 = fh.get_dict_path_l2(root_path)
    print(f"Dict Path L2: {dict_path_l2}")

    modified_path_list = fh.get_modified_path(index_folder_list, 'CT')
    print(f"Modified Path List: {modified_path_list}")

    # Example usage of image processing functions
    ct_path = "./data/ct/"
    nm_path = "./data/nm/nm.dcm"

    ct_3d_object = ip.get_ct_3d_object(ct_path)
    print(f"CT 3D Object Shape: {ct_3d_object.shape}")

    nm_3d_object = ip.get_nm_3d_object(nm_path)
    print(f"NM 3D Object Shape: {nm_3d_object.shape}")

    # Assuming slices_to_remove and process_mvp_image functions are called as part of the process
    delete_slices = ip.slices_to_remove(ct_path, nm_path)
    print(f"Slices to Remove: {delete_slices}")

    # Example for process_mvp_image function
    mvp_image_path = "./data/mvp/mvp.dcm"
    ip.process_mvp_image(input_path=mvp_image_path, delete_slices=delete_slices)

if __name__ == "__main__":
    main()
