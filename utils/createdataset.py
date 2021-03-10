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
percentage = 0.0
PATH_FILES = "data/landcovernet"
inputs_folder = "dataset/inputs"
target_folder = "dataset/target"

if not os.path.exists(inputs_folder):
    os.mkdir(inputs_folder)

if not os.path.exists(target_folder):
    os.mkdir(target_folder)

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

def create_level2(array):
    ar = array.copy()
    ar[np.where(array==3)]=2
    ar[np.where(array==4)]=3
    ar[np.where(array==5)]=4
    ar[np.where(array==6)]=5
    ar[np.where(array==7)]=5
    return ar

def create_level1(array):
    ar = array.copy()
    ar[np.where((array<=4) & (array>0))]=1
    ar[np.where(array>4)]=2
    return ar

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
                # b4 = gdal.Open(return_fn(p,"B05_10m.tif")).ReadAsArray()
                # b5 = gdal.Open(return_fn(p,"B06_10m.tif")).ReadAsArray()
                # b6 = gdal.Open(return_fn(p,"B07_10m.tif")).ReadAsArray()
                # b7 = gdal.Open(return_fn(p,"B08_10m.tif")).ReadAsArray()
                # b8 = gdal.Open(return_fn(p,"B11_10m.tif")).ReadAsArray()
                # b9 = gdal.Open(return_fn(p,"B12_10m.tif")).ReadAsArray()
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
                    # label_cp = create_level1(label_cp)
                    # f, axarr = plt.subplots(1,2)
                    # axarr[0].set_title("Original")
                    # axarr[1].set_title("Label")
                    # axarr[0].imshow(cv2.merge([normalize(b3),normalize(b2),normalize(b1)]))
                    # axarr[1].imshow(label_cp)
                    # plt.show()
                    # ask = input ("Use this image? R(Y to yes, any other to no)= ")
                    # if ask=="Y" or ask=="y":
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

