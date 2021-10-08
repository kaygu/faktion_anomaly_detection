from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams.update({'font.size': 18})

A = imread("Images/normal_dice/0/0.jpg", 0)
B = imread("Images/normal_dice/0/0.jpg",0)
C = imread("Images/anomalous_dice/img_17738_cropped.jpg",0)

print(np.shape(B))

B.save
