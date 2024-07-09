import os

def get_folder_list(root_path=".\\data\\", sub_group=0):
    """
    Get a list of folder paths containing specific sub-group names.
    
    Parameters:
    root_path (str): The root directory to search for folders. (".\\data\\")
    sub_group (int): The index of the sub-group to filter folders by.

    Returns:
    list: A sorted list of directory paths.
    """
    sub_group_list = ["_CT_", ".MVP.Planar", ".TA_", "inputData", "labelData", "resizedCTdcm", "resizedCTnii", "segData"]
    temp_list = os.walk(root_path)
    dir_names = []
    file_names = []
    for dir_name, _, file_name in temp_list:
        if sub_group_list[sub_group] in dir_name:
            dir_names.append(dir_name)
            file_names.append(file_name)
    return sorted(dir_names)

def get_index_folder_list(root_path=".\\data\\"):
    """
    Get a list of folder paths at the second level of the directory structure.
    
    Parameters:
    root_path (str): The root directory to search for folders.(".\\data\\")

    Returns:
    list: A list of directory paths.
    """
    folder_list = get_folder_list(root_path=root_path)
    index_folder_list = []
    for folder in folder_list:
        index_folder_list.append(os.path.dirname(folder))
    return index_folder_list

def get_index_folder_list_from_dict(input_dict):
    """
    Get a list of folder paths for CT, NM, and MVP from a nested dictionary structure.
    
    Parameters:
    input_dict (dict): The nested dictionary containing paths for CT, NM, and MVP.

    Returns:
    list: A list of lists, where each sublist contains paths for CT, NM, and MVP.
    """
    ret_list = []
    for key1 in input_dict:
        for key2 in input_dict[key1]:
            for key3 in input_dict[key1][key2]:
                ct_path = input_dict[key1][key2][key3]["CT"]
                nm_path = input_dict[key1][key2][key3]["NM"] + "\\" + os.listdir(input_dict[key1][key2][key3]["NM"])[0]
                mvp_path = input_dict[key1][key2][key3]["MVP"] + "\\" + os.listdir(input_dict[key1][key2][key3]["MVP"])[0]
                ret_list.append([ct_path, nm_path, mvp_path])
    return ret_list

def get_dict_path_l3(input_path=".\\data"):
    """
    Get a nested dictionary of folder paths at three levels of the directory structure.
    
    Parameters:
    input_path (str): The root directory to search for folders.(".\\data\\")

    Returns:
    dict: A nested dictionary containing folder paths at three levels.
    """
    folder_dict = {}
    root_path = input_path
    level1_folders = sorted(os.listdir(root_path + "\\"))  # [2306, 2307]
    for folder1 in level1_folders:
        level1_root_path = root_path + "\\" + folder1 + "\\"
        level2_folders = sorted(os.listdir(level1_root_path))  # list obj
        level2_dict = {}
        for folder2 in level2_folders:
            level2_root_path = level1_root_path + folder2 + "\\"
            level3_folders = sorted(os.listdir(level2_root_path))
            level3_dict = {}
            for folder3 in level3_folders:
                level3_root_path = level2_root_path + folder3 + "\\"
                level4_folders = sorted(os.listdir(level3_root_path))
                level4_dict = {}
                for folder4 in level4_folders:
                    if "CT_20" in folder4:
                        level4_dict["CT"] = level3_root_path + folder4 + "\\"
                    elif "TA_n" in folder4:
                        level4_dict["NM"] = level3_root_path + folder4
                    elif "MVP.P" in folder4:
                        level4_dict["MVP"] = level3_root_path + folder4
                level3_dict[folder3] = level4_dict
            level2_dict[folder2] = level3_dict
        folder_dict[folder1] = level2_dict
    return folder_dict

def get_dict_path_l2(input_path=".\\data\\"):
    """
    Get a nested dictionary of folder paths at two levels of the directory structure.
    
    Parameters:
    input_path (str): The root directory to search for folders.(".\\data\\")

    Returns:
    dict: A nested dictionary containing folder paths at two levels.
    """
    root_path = input_path
    level1_folders = sorted(os.listdir(root_path))
    folder_dict = {}
    for folder1 in level1_folders:
        level2_folders = sorted(os.listdir(os.path.join(root_path, folder1)))
        level2_dict = {}
        for folder2 in level2_folders:
            level3_folders = sorted(os.listdir(os.path.join(root_path, folder1, folder2)))
            level2_dict[folder2] = level3_folders
        folder_dict[folder1] = level2_dict
    return folder_dict

def get_modified_path(input_list=None, sub_group=None):
    """
    Get a list of modified paths based on a specific sub-group filter.
    
    Parameters:
    input_list (list): The list of input folder paths.
    sub_group (str): The sub-group to filter paths by.

    Returns:
    list: A list of filtered and modified folder paths.
    """
    if input_list is None:
        root_path = get_index_folder_list()
    else:
        root_path = input_list
    ret_list = []
    filter_dict = {"CT": "CT_20", "NM": "TA_n", "MVP": "MVP.P", "RESCT": "_resizedCTdcm", "SEG": "_segData", "NIFTICT": "resizedCTnii"}
    try:
        if sub_group in filter_dict.keys():
            filter_str = filter_dict[sub_group]
        for folder in root_path:
            sub_folders = os.listdir(folder)
            for sub_folder in sub_folders:
                if filter_str in sub_folder:
                    ret_path = os.path.join(folder, sub_folder)
                    ret_list.append(ret_path)
        return ret_list
    except Exception as e:
        print(f"Error in sub_group: {e}")

def create_subfolders(input_path):
    """
    Create subfolders within the specified input path.
    
    Parameters:
    input_path (str): The path where subfolders should be created.

    Returns:
    None
    """
    idx = input_path.split("\\")[-1]
    os.mkdir(os.path.join(input_path, idx + "_resizedCTdcm"))
    os.mkdir(os.path.join(input_path, idx + "_resizedCTnii"))
    os.mkdir(os.path.join(input_path, idx + "_segData"))
    os.mkdir(os.path.join(input_path, idx + "_inputData"))
    os.mkdir(os.path.join(input_path, idx + "_labelData"))

def show_results(init_key, last_key, len_ct, len_nm, skipped_dict):
    """
    Display the results of the processing.
    
    Parameters:
    init_key (int): Initial key for NM.
    last_key (int): Final key for NM.
    len_ct (int): Number of CT slices.
    len_nm (int): Number of NM slices.
    skipped_dict (dict): Dictionary of skipped NM slices.

    Returns:
    None
    """
    print(f"Initial NM Value: {init_key}")
    print(f"Final NM Slide Number: {last_key}")
    print(f"Number of CT Slices: {len_ct}")
    print(f"Number of NM Slices: {len_nm}")
    print(f"Skipped NM Slice Numbers: {skipped_dict}")
