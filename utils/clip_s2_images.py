from osgeo import gdal
import os
import pathlib
from tqdm import tqdm
import numpy as np
from cv2 import cv2 
import rasterio
from shapely.geometry import Polygon
from rasterio.mask import mask
import shapely 



path = '/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/data'
outpath = '/home/felipe/hdd/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test'

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
    return str(list(pathlib.Path(path).rglob(f'*{ends}.jp2'))[0])

def get_bounds(img,n):
    ulx, xres, _, uly, _, yres  = img.GetGeoTransform()
    lrx = ulx + (img.RasterXSize * xres)
    lry = uly + (img.RasterYSize * yres)
    
    stepx = (lrx-ulx)/(n)
    stepy = (lry-uly)/(n)

    x = np.arange(ulx,lrx+1,stepx)
    y = np.arange(uly,lry-1,stepy)

    geo = []
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            p = Polygon([[x[i],y[j]], [x[i],y[j+1]], [x[i+1],y[j+1]], [x[i+1],y[j]]])
            geo.append(p)

    return geo

def clip(img,wkt,output):
    with rasterio.open(img) as src:
        out_image, out_transform = mask(src, [wkt], crop=True)
        out = out_image[out_image!=0]
        if len(out)==0:
            return
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

    with rasterio.open(output, "w", **out_meta) as output:
        output.write(out_image)
    
for i in tqdm(os.listdir(path)):
    p = os.path.join(path,i)
    b2 = gdal.Open(find_band(p,'B02_10m'))
    b3 = gdal.Open(find_band(p,'B03_10m'))
    b4 = gdal.Open(find_band(p,'B04_10m'))
    b8 = gdal.Open(find_band(p,'B08_10m'))
    array = [j.ReadAsArray() for j in [b2,b3,b4,b8]]
    src = os.path.join(outpath,i+".tif")
    create_img(src,b2,array)

    img = gdal.Open(src)
    geo = get_bounds(img,10)
    for n,j in enumerate(geo):
        output = os.path.splitext(src)[0]+f"_{n}.tif"
        clip(src,j,output)

    os.remove(src)



