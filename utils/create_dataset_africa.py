import shutil
import os
import math
from osgeo import gdal
import rasterio 
from rasterio.mask import mask
from shapely.geometry import Polygon
from shapely.ops import transform
from tqdm import tqdm
import numpy as np
import pyproj

path = 'datasets/Africa'
output = 'data'
length = 256

f = os.path.join(output,'target2')
if not os.path.exists(f):
    os.mkdir(f)
f = os.path.join(output,'inputs')
if not os.path.exists(f):
    os.mkdir(f)
f = os.path.join(output,'target')
if not os.path.exists(f):
    os.mkdir(f)

def change_labels(array):
    ar = array.copy()
    ar[np.where(array==200)]=11
    return ar

def create_bounds(vmin,vmax,length,scale = 10):
    rang = vmax-vmin
    length *= scale
    n = math.ceil(rang/length)
    values = []
    v = vmin
    for i in range(1,n+1):
        if i==n:
            vv = (v+length)-vmax
            values.append([v-vv,vmax])
        else:
            values.append([v,v+length])
            v = (v+length) - (rang*0.00005)
    #     print(values[-1])
    # print("#"*50)
    return values


def create_img(output,img,array,dtype = gdal.GDT_UInt16):
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,img.RasterXSize,img.RasterYSize,len(array),dtype)
    for i in range(len(array)):
        b = dst.GetRasterBand(i+1)
        b.WriteArray(array[i])
    dst.SetProjection(img.GetProjection())
    dst.SetGeoTransform(img.GetGeoTransform())
    dst.FlushCache()


def clip(img,wkt,output):
    with rasterio.open(img) as src:
        # print(wkt)
        # print(gdal.Open(img).GetGeoTransform())
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
    if i.endswith('label.tif'):
        label = os.path.join(path,i)
        inp = os.path.join(path,i[0:-10]+".tif")

        img = gdal.Open(label)
        vmin_x, xres, _, vmax_y, _, yres  = img.GetGeoTransform()
        vmax_x = vmin_x + (img.RasterXSize * xres)
        vmin_y = vmax_y + (img.RasterYSize * yres)
        boundsx = create_bounds(vmin_x,vmax_x,length,xres)
        boundsy = create_bounds(vmin_y,vmax_y,length,xres)

        geo = []
        for m in boundsx:
            for j in boundsy:
                xmax,xmin = m
                ymax,ymin = j
                p = Polygon([[xmin,ymax], 
                            [xmax,ymax], 
                            [xmax,ymin], 
                            [xmin,ymin]])
                geo.append(p)

        project = pyproj.Transformer.from_proj(
                    pyproj.Proj(img.GetProjection()),
                    pyproj.Proj(gdal.Open(inp).GetProjection()),
                    always_xy=True)
            
        geo_inp = []
        for polygon in geo:
            geo_inp.append(transform(project.transform, polygon))
        
        for wkt in range(len(geo)):
            n = wkt+1
            name = os.path.splitext(i)[0]+f'_{n}.tif'
            output_label = os.path.join(output,'target2',name)
            output_input = os.path.join(output,'inputs',name)
            clip(label,geo[wkt],output_label)
            if os.path.exists(output_label):
                clip(inp,geo_inp[wkt],output_input)
                

outpath = os.path.join(output,'target')
inp = os.path.join(output,'inputs')
for i in tqdm(os.listdir(inp)):
    f = os.path.join(output,'target2',i)
    # if not os.path.exists(f):
    #     os.remove(os.path.join(inp,i))
    # else:
    try:
        img = gdal.Open(f)
        array = img.ReadAsArray()
        array = change_labels(array)
        out = os.path.join(outpath,i)
        create_img(out,img,[array],gdal.GDT_Byte)
    except AttributeError:
        os.remove(os.path.join(inp,i))

    