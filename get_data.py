import os
import cv2
import numpy as np

def get_JPGs(dir: str) -> list:
    """
    Builds a list of JPG files from a given directory and every sub directory
    :param dir: Directory where to search for jpg files
    :return: List of JPG files
    """
    jpg_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                jpg_list.append(root + os.sep + file)
    return jpg_list

def get_matrices(files: list) -> list:
    matrices = []
    for file in files:
      matrices.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    return matrices


def get_data():
    # Get dices
    defect_dice_files = get_JPGs('data/anomalous_dice')
    normal_dices_files = get_JPGs('data/normal_dice')
    defect_matrices = np.asarray(get_matrices(defect_dice_files))
    normal_matrices = np.asarray(get_matrices(normal_dices_files))

    
    preprocessed_d = defect_matrices.reshape(-1,128,128, 1)
    preprocessed_n = normal_matrices.reshape(-1,128,128, 1)

    x = np.concatenate([preprocessed_n, preprocessed_d])
    y = np.concatenate([[0 for _ in range(preprocessed_n.shape[0])], [1 for _ in range(preprocessed_d.shape[0])]])

    print(x.shape)

    return x, y