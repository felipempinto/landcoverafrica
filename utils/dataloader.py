import torch
# from torch.utils import data
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

def get_file_names(path):
    return [os.path.join(path,i) for i in os.listdir(path)]

def normalize(array):
    array = array/255#10000
    return np.array(array,dtype=np.float32)

def change(array):
    im = rgb2gray(array)*255
    im = np.array(im,dtype=np.uint8)
    im2 = im.copy()
    im2[im==18]=1
    im2[im==72]=2
    im2[im==182]=3
    im2[im==200]=4
    im2[im==236]=5
    im2[im==255]=6
    # unique = np.unique(im2)
    return im2
    
    

class DataSet(Dataset):
    def __init__(self, inputs,targets,use_cache=False):
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache=use_cache

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for _, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                x = gdal.Open(img_name).ReadAsArray()
                y = gdal.Open(tar_name).ReadAsArray()
                x = normalize(x)

                self.cached_data.append((x, y))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,index):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            x = gdal.Open(input_ID).ReadAsArray()
            y = gdal.Open(target_ID).ReadAsArray()
            x = normalize(x)

        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)

        # x = normalize(x)
        # x = torch.from_numpy(x).type(self.inputs_dtype)
        # y = torch.from_numpy(y).type(self.targets_dtype)

        return x,y


if __name__=="__main__":

    BATCH_SIZE = 10
    X = get_file_names('dataset/inputs')
    y = get_file_names('dataset/target')

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.25,shuffle=True)

    dataset_train = DataSet(inputs = X_train, targets = y_train)
    dataset_valid = DataSet(inputs = X_test, targets = y_test)

    dataloader_training = DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True)
    dataloader_validation = DataLoader(dataset=dataset_valid,batch_size=BATCH_SIZE,shuffle=True)

    x, y = next(iter(dataloader_training))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')