import numpy as np
import os
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt

# path = "/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/Datasets/Harald/landcovernet_50/ref_landcovernet_v1_labels_29PKL_00/2018_01_29"
# path = "/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/Datasets/Harald/landcovernet_50/ref_landcovernet_v1_labels_29PKL_00/2018_01_04"

path = "/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/Datasets/Harald/landcovernet_50/ref_landcovernet_v1_labels_29PKL_00"

for m in os.listdir(path):
    # print(m)
    p = os.path.join(path,m)
    if os.path.isdir(p):
        b4 = gdal.Open(os.path.join(p,[i for i in os.listdir(p) if i[18:21]=="B04"][0])).ReadAsArray()
        b3 = gdal.Open(os.path.join(p,[i for i in os.listdir(p) if i[18:21]=="B03"][0])).ReadAsArray()
        b2 = gdal.Open(os.path.join(p,[i for i in os.listdir(p) if i[18:21]=="B02"][0])).ReadAsArray()
        b4 = b4/np.max(b4)
        b3 = b3/np.max(b3)
        b2 = b2/np.max(b2)
        c = cv2.merge((b4,b3,b2))
        cloud = gdal.Open(os.path.join(p,[i for i in os.listdir(p) if i[18:21]=="CLD"][0])).ReadAsArray()
        print(np.sum(cloud)/(np.shape(cloud)[0]*np.shape(cloud)[1]))
        plt.imshow(c)
        plt.show()
        # plt.imshow(cloud)
        # plt.show()