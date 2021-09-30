import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
    for file in files[1:]:
        matrices.append(cv2.imread(file))
    return matrices

if __name__ == "__main__":
    print('Hello World!')
    # Get dices
    anomalous_dice_files = get_JPGs('data/anomalous_dice')
    normal_dices_files = get_JPGs('data/normal_dice')
    anomalous_matrices = get_matrices(anomalous_dice_files)
    normal_matrices = get_matrices(normal_dices_files)
    print(len(anomalous_matrices))
    print(len(normal_matrices))
    # np.save("file.npy", matrices)
