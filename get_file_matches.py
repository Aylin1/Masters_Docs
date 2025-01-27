import os
import re

def find_rgbi_image(laz_data_path, dop_folder):
    file_identifier = os.path.basename(laz_data_path).split('.')[0].split('_')[-1]
    for root, _, files in os.walk(dop_folder):
        for file in files:
            if file_identifier in file and file.endswith('.tif'):
                return os.path.join(root, file)
    return None

def get_als_dop_matches(als_folder, dop_folder):
    matches = {}
    for root, _, files in os.walk(als_folder):
        for file in files:
            if file.endswith('.laz'):
                als_file_path = os.path.join(root, file)
                dop_image_path = find_rgbi_image(als_file_path, dop_folder)
                if dop_image_path:
                    matches[als_file_path] = dop_image_path
    return matches

def get_als_tif_matches(root_dir, als_subfolder="als", tif_subfolder="ground_truth_masks/tree_masks", identifier_pattern=r'\d{5}-\d{4}', contains=None):
    """
    Match ALS (.laz) files with corresponding TIF images using a common 5-digit-4-digit identifier.
    The identifier is extracted from the ALS file name or folder name, and the function searches
    for subfolder names and file names that contain the identifier.
    
    Parameters:
    root_dir (str): Root directory containing all subfolders.
    als_subfolder (str): Subdirectory containing ALS files.
    tif_subfolder (str): Subdirectory containing TIF ground truth masks.
    identifier_pattern (str, optional): Regex pattern to match 5-digit-4-digit format (default: '\d{5}-\d{4}').
    contains (str, optional): Substring to search for in the TIF file names (default: None, no filter applied).

    Returns:
    dict: A dictionary where keys are ALS file paths and values are the matched TIF file paths.
    """
    matches = {}

    als_folder = os.path.join(root_dir, als_subfolder)
    tif_folder = os.path.join(root_dir, tif_subfolder)

    # Loop through ALS files in the ALS folder
    for root, _, files in os.walk(als_folder):
        for file in files:
            if file.endswith('.laz'):
                # Get the full path to the ALS file
                als_file_path = os.path.normpath(os.path.join(root, file))

                # Extract the unique identifier from the ALS file name or folder name
                identifier = extract_file_identifier(als_file_path, identifier_pattern)
                
                if identifier is None:
                    print(f"Identifier not found for ALS file: {als_file_path}")
                    continue
                
                matched_tif_path = None

                # Look for the identifier in subfolders and filenames in the TIF folder
                for tif_root, _, tif_files in os.walk(tif_folder):
                    # Check if the identifier is in the subfolder name
                    if identifier in os.path.basename(tif_root):
                        for tif_file in tif_files:
                            if tif_file.endswith('.tif'):
                                if contains is None or contains in tif_file:
                                    matched_tif_path = os.path.normpath(os.path.join(tif_root, tif_file))
                                    break

                    # Check if the identifier is in the TIF file name directly
                    for tif_file in tif_files:
                        if identifier in tif_file and tif_file.endswith('.tif'):
                            if contains is None or contains in tif_file:
                                matched_tif_path = os.path.normpath(os.path.join(tif_root, tif_file))
                                break

                    if matched_tif_path:
                        break

                # Store the match if a corresponding TIF file is found
                if matched_tif_path:
                    matches[als_file_path] = matched_tif_path  # Store the full path as a single string
                    print(f"✅ ALS File: {als_file_path} -> Matched TIF: {matched_tif_path}")
                    print("")
                else:
                    print(f"❌ No match found for identifier {identifier} for ALS file: {als_file_path}")

    return matches


def extract_file_identifier(file_path, identifier_pattern):
    """
    Extract the unique 5-digit-4-digit identifier from a file path or its parent folder name.
    """
    # Extract identifier from the parent folder name
    folder_name = os.path.basename(os.path.dirname(file_path))
    identifier_from_folder = re.search(identifier_pattern, folder_name)

    # Extract identifier from the file name
    file_name = os.path.basename(file_path)
    identifier_from_file = re.search(identifier_pattern, file_name)
    
    if identifier_from_folder:
        print(f"📂 Extracted identifier '{identifier_from_folder.group()}' from folder: {folder_name}")
        return identifier_from_folder.group()
    elif identifier_from_file:
        print(f"📄 Extracted identifier '{identifier_from_file.group()}' from file: {file_name}")
        return identifier_from_file.group()
    else:
        print(f"❌ Could not extract identifier from: {file_path}")
        return None



def get_tif_file_matches(root_dir, folder1, folder2, identifier_pattern=r'\d{5}-\d{4}', contains1=None, contains2=None):
    """
    Match TIF files from any two folders within a root directory using a 5-digit-4-digit identifier.
    
    Parameters:
    root_dir (str): The root directory containing folder1 and folder2.
    folder1 (str): The subfolder name where the first set of TIF files is located.
    folder2 (str): The subfolder name where the second set of TIF files is located.
    identifier_pattern (str): Regex pattern to extract identifiers (default: 5-digit-4-digit format).
    contains1 (str, optional): Substring filter for files in folder1 (default: None).
    contains2 (str, optional): Substring filter for files in folder2 (default: None).

    Returns:
    dict: A dictionary where keys are file paths from folder1 and values are matched file paths from folder2.
    """
    matches = {}
    folder1_path = os.path.join(root_dir, folder1)
    folder2_path = os.path.join(root_dir, folder2)

    print(f"Folder1 Path: {folder1_path}")
    print(f"Folder2 Path: {folder2_path}")

    # Iterate through all TIF files in folder1
    for root1, _, files1 in os.walk(folder1_path):
        for file1 in files1:
            if not file1.endswith('.tif') or (contains1 and contains1 not in file1):
                continue

            tif1_path = os.path.normpath(os.path.join(root1, file1))
            identifier = re.search(identifier_pattern, file1)
            if not identifier:
                print(f"❌ No identifier found in file: {file1}")
                continue

            identifier = identifier.group()

            matched_tif2_path = None

            # Search for matches in folder2
            for root2, subdirs, files2 in os.walk(folder2_path):
                # Check subfolder names
                if identifier in os.path.basename(root2):
                    for file2 in files2:
                        # Check if the file name contains the given substring (if specified)
                        if file2.endswith('.tif') and (contains2 is None or contains2 in file2):
                            matched_tif2_path = os.path.normpath(os.path.join(root2, file2))
                            break

                # Check filenames in folder2
                for file2 in files2:
                    if file2.endswith('.tif') and identifier in file2:
                        if contains2 is None or contains2 in file2:
                            matched_tif2_path = os.path.normpath(os.path.join(root2, file2))
                            break

                if matched_tif2_path:
                    break

            if matched_tif2_path:
                matches[tif1_path] = matched_tif2_path
            else:
                print(f"❌ No match found for identifier {identifier} in {tif1_path}")

    return matches
