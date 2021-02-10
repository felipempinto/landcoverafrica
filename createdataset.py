from cv2 import cv2
from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
gdal.UseExceptions()

# Registering errors
logging.basicConfig(filename='createdataset.log')

# Using landcovernet dataset to create the dataset
#
# This first parameter will be used to select the percentage we can accept to our model,
# the range of this variable is from 0.0 to 100.0
percentage = 0.01
PATH_FILES = "data/landcovernet"
inputs_folder = "dataset/inputs"
target_folder = "dataset/target"

def create_tif(img,array,output,dtype = gdal.GDT_Float32):
    im = gdal.Open(img)
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,im.RasterXSize,im.RasterYSize,len(array),dtype)
    for i in range(len(array)):
        b = dst.GetRasterBand(i+1)
        b.WriteArray(array[i])
    dst.SetProjection(im.GetProjection())
    dst.SetGeoTransform(im.GetGeoTransform())
    dst.FlushCache()


def return_fn(path,ends):
    return os.path.join(path,[f for f in os.listdir(path) if f.endswith(ends)][0])

def normalize(array):
    maximum = np.max(array)
    array = array/maximum
    return np.array(array,dtype=np.float32)

datasets = {}
c = 0

for i in tqdm(os.listdir(PATH_FILES)):
    datasets[i]=0
    cont=0
    path = os.path.join(PATH_FILES,i)
    dates = [j for j in os.listdir(path) if os.path.isdir(os.path.join(path,j))]
    try:
        y = os.path.join(path,[j for j in os.listdir(path) if j.endswith('.tif')][0])
    except IndexError as e:
        logging.error(f'IndexError found: "{e}", the Ground Truth file for the path {i} does not exists')
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
            except IndexError as e:
                logging.error(f'IndexError found: "{e}", bands not found in the {p} folder')
            except RuntimeError as e:
                logging.error(f'RuntimeError found: "{e}", some of the files have problems to be opened')
            else:
                array = [b3,b2,b1]
                pct = np.sum(cl)/(np.shape(cl)[0]*np.shape(cl)[1])
                if pct<=percentage:
                    ar = cv2.merge(array)
                    arr = 1+(np.all(ar[:,:]==np.array([0,0,0]),axis=2)*-1)
                    label_cp = label*arr
                    label_out = os.path.join(target_folder,f'{i}_{datasets[i]}.tif')
                    create_tif(return_fn(p,"B02_10m.tif"),[label_cp],label_out,dtype = gdal.GDT_Byte)
                    output = os.path.join(inputs_folder,f'{i}_{datasets[i]}.tif')
                    create_tif(return_fn(p,"B02_10m.tif"),array,output,dtype = gdal.GDT_UInt16)
                    datasets[i]+=1
                    c+=1

for i in datasets.keys():
    if datasets[i]>0:
        print(f'{i} = {round((datasets[i]/c)*100,4)}%, total = {datasets[i]}')
print(f'Total amount of images: {c}')

