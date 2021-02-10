from osgeo import gdal
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

file = '/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/nn/landcoverafrica/dataset/inputs/ref_landcovernet_v1_labels_28QDE_09_115.tif'
label = '/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/nn/landcoverafrica/dataset/target/ref_landcovernet_v1_labels_28QDE_09.tif'

im = gdal.Open(file)
lb = gdal.Open(label)

array1 = im.ReadAsArray()
array2 = lb.ReadAsArray()
ar = cv2.merge(array1)/10000

arr = 1+(np.all(ar[:,:]==np.array([0,0,0]),axis=2)*-1)

plt.imshow(arr)
plt.show()

