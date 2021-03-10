import shutil
import os
import math
from osgeo import gdal
import rasterio 
from rasterio.mask import mask
from shapely.geometry import Polygon
from tqdm import tqdm

labels = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test/2-km/post/labels'
inputs = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test'
output = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/nn/landcoverafrica/dataset'
length = 256

def create_bounds(vmin,vmax,length):
    rang = vmax-vmin
    length *=10
    n = math.ceil(rang/length)
    values = []
    v = vmin
    c = (rang%length)*2
    for i in range(1,n+1):        
        if i==n:
            vv = (v+length)-vmax
            values.append([v-vv,vmax])
        else:
            values.append([v,v+length])
            v = (v+(i*length))-(c*i)
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
    

for i in tqdm(os.listdir(labels)):
    label = os.path.join(labels,i)
    inp = os.path.join(inputs,i)

    img = gdal.Open(label)

    vmin_x, xres, _, vmax_y, _, yres  = img.GetGeoTransform()
    vmax_x = vmin_x + (img.RasterXSize * xres)
    vmin_y = vmax_y + (img.RasterYSize * yres)
    
    boundsx = create_bounds(vmin_x,vmax_x,length)
    boundsy = create_bounds(vmin_y,vmax_y,length)

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

    for n,wkt in enumerate(geo,1):
        output_label = os.path.join(output,'target',os.path.splitext(i)[0]+f'_{n}.tif')
        output_input = os.path.join(output,'inputs',os.path.splitext(i)[0]+f'_{n}.tif')
        clip(label,wkt,output_label)
        clip(inp,wkt,output_input)


