import cv2
from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
gdal.UseExceptions()

# Using landcovernet dataset to create the dataset
PATH_FILES = "/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/Datasets/Harald/landcovernet_50"
OUTPATH = os.path.join(os.path.dirname(__file__),"data")
if not os.path.exists(OUTPATH):
    os.mkdir(OUTPATH)

def return_fn(path,ends):
    return os.path.join(path,[f for f in os.listdir(path) if f.endswith(ends)][0])

def normalize(array):
    maximum = np.max(array)
    array = array/maximum
    return array

datasets = []
c = 0
for i in os.listdir(PATH_FILES):
    path = os.path.join(PATH_FILES,i)
    dates = [j for j in os.listdir(path) if os.path.isdir(os.path.join(path,j))]
    try:
        y = os.path.join(path,[j for j in os.listdir(path) if j.endswith('.tif')][0])
    except IndexError:
        print(f"{i} has no labels")
    else:
        l = gdal.Open(y)
        label = l.GetRasterBand(1).ReadAsArray()
        for date in dates:
            p = os.path.join(path,date)
            try:
                b1 = gdal.Open(return_fn(p,"B02_10m.tif")).ReadAsArray()
                b2 = gdal.Open(return_fn(p,"B03_10m.tif")).ReadAsArray()
                b3 = gdal.Open(return_fn(p,"B04_10m.tif")).ReadAsArray()
                cl = gdal.Open(return_fn(p,"CLD_10m.tif")).ReadAsArray()
            except IndexError:
                print(f"date: {date} i {i} doesn't have the bands")
            else:
                merged = np.array([b1,b2,b3])
                pct = np.sum(cl)/(np.shape(cl)[0]*np.shape(cl)[1])
                if pct<=0.01:
                    datasets.append([merged,label])
                    c+=1
print(c)

np.random.shuffle(datasets)
np.save("training_data_clouds_001.npy", datasets)



