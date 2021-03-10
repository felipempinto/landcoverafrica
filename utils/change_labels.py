from osgeo import gdal, ogr, osr
import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import cv2
import tkinter as tk
from tkinter import ttk
import random

gdal.UseExceptions()

original = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test'
# path = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test/1-supervised/results'
# output_path = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test/1-supervised/results/post'
path = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test/2-km'
output_path = '/media/felipe/3dbf30eb-9bce-46d8-a833-ec990ba72625/Documentos/Empresa/Upwork/Harald/Project1/Datasets/createS2/2a/third_test/2-km/post'

output_path_lb = os.path.join(output_path,'labels')
output_path_sh = os.path.join(output_path,'shapes')

def normalize(array,bands=''):
    final = []
    if bands=='':
        bands = [i for i in range(1,len(array)+1)]
    for i in bands:
        ar = array[i-1]/array[i-1].max()
        final.append(ar)
    return final

def create_img(img,array,output,dtype=gdal.GDT_Byte):
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,img.RasterXSize,img.RasterYSize,len(array),dtype)
    for n,ar in enumerate(array,1):
        b = dst.GetRasterBand(n)
        b.WriteArray(ar)
    dst.SetProjection(img.GetProjection())
    dst.SetGeoTransform(img.GetGeoTransform())
    dst.FlushCache()

def polygonize(img,shp_dst):
    srs = osr.SpatialReference()
    src_ds = gdal.Open(img)
    srs.ImportFromWkt(src_ds.GetProjection())
    srcband = src_ds.GetRasterBand(1)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(shp_dst)
    dst_layer = dst_ds.CreateLayer(shp_dst, srs = srs)
    newField = ogr.FieldDefn('id', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    gdal.Polygonize(srcband, None, dst_layer, 0, [],callback=None )

def change_label(array,dic):
    ar = array.copy()
    for i in dic.keys():
        ar[np.where(array==i)]=dic[i]
    return ar

lista = list(set([i[0:-8]+'.tif' for i in os.listdir(path) if i.endswith('.tif')]))
random.shuffle(lista)

f = open(os.path.join(output_path,"done.txt"))
remove = [i.replace('\n','') for i in f]
# print(len(lista))
lista = [i for i in lista if i not in remove]
# print(len(lista))

for i in lista:
    f = open(os.path.join(output_path,"done.txt"),mode="a")
    p = os.path.join(path,i)
    img = os.path.join(original,i)
    im = gdal.Open(img)
    array = im.ReadAsArray()
    array_im = normalize(array,bands=[4,2,1])
    array_im = cv2.merge(array_im)
    array_im_rgb = normalize(array,bands=[3,2,1])
    array_im_rgb = cv2.merge(array_im_rgb)

    p = os.path.splitext(p)[0]
    
    fig=plt.figure(figsize=(15, 15))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(array_im)
    ax1.set_title("Original Image (NIR-G-B")

    ax = fig.add_subplot(2, 3, 2,sharex=ax1,sharey=ax1)
    ax.imshow(array_im_rgb)
    ax.set_title("Original Image (R-G-B)")

    c=3
    for j in [3,4,5,7]:
        label = p + f'_KM{j}.tif'
        lb = gdal.Open(label)
        array_lb = lb.ReadAsArray()
        uniques = np.unique(array_lb)

        ax2 = fig.add_subplot(2, 3, c,sharex=ax1,sharey=ax1)
        ax2.imshow(array_lb)
        ax2.set_title(f"Original Label N: {j}")
        c+=1
    plt.show()
    
    try:
        selected = int(input("Select one image to continue: "))
        label = p + f'_KM{selected}.tif'
        lb = gdal.Open(label)
        array_lb = lb.ReadAsArray()
        uniques = np.unique(array_lb)
    except (RuntimeError,ValueError):
        continue

    array_lb_ft3 = cv2.medianBlur(array_lb,3)
    array_lb_ft5 = cv2.medianBlur(array_lb,5)

    root = tk.Tk()
    for n in range(len(uniques)):
        tk.Label(root,text=f'Label {n}').pack(padx=5,pady=5)
        tk.Entry(root).pack(padx=5,pady=5)

    fig=plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(array_lb)
    ax1.set_title(f"(1) Original Label N: {len(np.unique(array_lb))}")
    ax2 = fig.add_subplot(2, 2, 2,sharex=ax1,sharey=ax1)
    ax2.imshow(array_im)
    ax2.set_title("Original Image")
    ax3 = fig.add_subplot(2, 2, 3,sharex=ax1,sharey=ax1)
    ax3.imshow(array_lb_ft3)
    ax3.set_title(f"(2) Filtered 3 N: {len(np.unique(array_lb_ft3))}")
    ax4 = fig.add_subplot(2, 2, 4,sharex=ax1,sharey=ax1)
    ax4.imshow(array_lb_ft5)
    ax4.set_title(f"(3) Filtered 5 N: {len(np.unique(array_lb_ft5))}")
    plt.show()

    selected = input('Select the image do you wish to use (1,2 or 3): ')

    dic = {}
    for j in uniques:
        value = input(f"The new label for the '{j}' value: ")
        dic[j]=value

    try:
        if int(selected)==1:
            array_lb = change_label(array_lb,dic)
        elif int(selected)==2:
            array_lb = change_label(array_lb_ft3,dic)
        elif int(selected)==3:
            array_lb = change_label(array_lb_ft5,dic)
        else:
            continue
    except ValueError:
        continue

    

    edit = input("Do you need to edit? R(y):")
    if edit=='y':
        shp_dst = os.path.join(output_path_sh,os.path.splitext(i)[0]+'.shp')
        polygonize(img,shp_dst)
        f.write(i+'\n')
    else:
        lb_out = os.path.join(output_path_lb,i)
        create_img(im,[array_lb],lb_out)
        f.write(i+'\n')
    f.close()

    # lb_ft3 = os.path.join(output_path,i[0:-8]+'_median3.tif')
    # shp_dst = os.path.splitext(lb_ft3)[0]+'.shp'
    # create_img(im,[array_lb_ft3],lb_ft3)
    # # polygonize(img,shp_dst)

    # lb_ft5 = os.path.join(output_path,i[0:-8]+'_median5.tif')
    # shp_dst = os.path.splitext(lb_ft5)[0]+'.shp'
    # create_img(im,[array_lb_ft5],lb_ft5)
    # polygonize(lb_ft5,shp_dst)
    


