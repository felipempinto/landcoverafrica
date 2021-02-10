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
percentage = 0.1
PATH_FILES = "data/landcovernet"
inputs_folder = "dataset/inputs"
target_folder = "dataset/target"

def return_fn(path,ends):
    return os.path.join(path,[f for f in os.listdir(path) if f.endswith(ends)][0])

def normalize(array):
    maximum = np.max(array)
    array = array/maximum#10000#
    return np.array(array,dtype=np.float32)

# datasets = []
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
        cv2.imwrite(os.path.join(target_folder,f'{i}.tif'),label)
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
                # merged = np.array([b1,b2,b3])
                merged = cv2.merge((normalize(b1),normalize(b2),normalize(b3)))
                pct = np.sum(cl)/(np.shape(cl)[0]*np.shape(cl)[1])
                if pct<=percentage:
                    cv2.imwrite(os.path.join(inputs_folder,f'{i}_{datasets[i]}.tif'),merged)
                    datasets[i]+=1
                    c+=1
                    # datasets.append([merged,label])

for i in datasets.keys():
    print(f'{i} = {round((datasets[i]/c)*100,4)}')
print(f'Total amount of images: {c}')

# np.random.shuffle(datasets)
# np.save("dataset/training_data.npy", datasets)



