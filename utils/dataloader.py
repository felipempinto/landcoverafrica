import torch
# from torch.utils import data
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.model_selection import train_test_split

def get_file_names(path):
    return [os.path.join(path,i) for i in os.listdir(path)]

def normalize(array):
    array = array/10000
    return np.array(array,dtype=np.float32)

class DataSet(Dataset):
    def __init__(self, inputs,targets):
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,index):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        x = gdal.Open(input_ID).ReadAsArray()
        x = normalize(x)
        y = gdal.Open(target_ID).ReadAsArray()

        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)

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