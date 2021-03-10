from osgeo import gdal
import os
import pathlib
from tqdm import tqdm
import numpy as np
from cv2 import cv2 

path = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test'
outpath = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test/km'
K = 7

def create_img(output,img,array,dtype = gdal.GDT_UInt16):
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,img.RasterXSize,img.RasterYSize,len(array),dtype)
    for i in range(len(array)):
        b = dst.GetRasterBand(i+1)
        b.WriteArray(array[i])
    dst.SetProjection(img.GetProjection())
    dst.SetGeoTransform(img.GetGeoTransform())
    dst.FlushCache()

def find_band(path,ends):
    return str(list(pathlib.Path(path).rglob(f'*{ends}.tif'))[0])
    
for i in tqdm(os.listdir(path)):
    if i.endswith('.tif'):
        out = os.path.join(outpath,os.path.splitext(i)[0]+f"_KM{K}.tif")

        if not os.path.exists(out):
            im = gdal.Open(os.path.join(path,i))
            array = im.ReadAsArray()
            img = cv2.merge(array)
            
            Z = img.reshape((-1,img.shape[2]))
            Z = np.float32(Z)
            Z = cv2.UMat(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            # label = label.ravel()
            label = cv2.UMat.get(label)
            res2 = [label.reshape((img.shape[0:-1]))]
            create_img(out,im,res2,dtype=gdal.GDT_Byte)





