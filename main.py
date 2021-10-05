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
      matrices.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    return matrices

if __name__ == "__main__":
    print('Hello World!')
    # Get dices
    anomalous_dice_files = get_JPGs('data/anomalous_dice')
    normal_dices_files = get_JPGs('data/normal_dice')
    anomalous_matrices = np.asarray(get_matrices(anomalous_dice_files))
    normal_matrices = np.asarray(get_matrices(normal_dices_files))
    print(anomalous_matrices[0].shape)
    print(anomalous_matrices.shape)

    plt.bar([0,1], [anomalous_matrices.shape[0], normal_matrices.shape[0]])
    plt.title('dataset sizes')
    plt.xticks([0,1], ['anomalous', 'normal'])
    plt.ylabel('number of samples');
    plt.show()


    def plot_raster(image):
        plt.imshow(image, cmap="gray")
        plt.axis('off')
        plt.show()

    dog_sample = anomalous_matrices[0].reshape(28,28)
    plot_raster(dog_sample)
    # preprocessed_a_d= anomalous_matrices.reshape(-1,28,28)
    # preprocessed_n_d = normal_matrices.reshape(-1,28,28)

    # # Normalizing
    # preprocessed_a_d = preprocessed_a_d/255
    # preprocessed_n_d = preprocessed_n_d/255



    # np.save("data/normal_dices.npy", normal_matrices)
    # np.save("data/anomal_dices.npy", anomalous_matrices)