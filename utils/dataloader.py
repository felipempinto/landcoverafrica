import torch
from torch.utils import data
from cv2 import cv2
import os
import numpy as np

class Dataset(data.Dataset):
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

        x = cv2.imread(input_ID)
        y = cv2.imread(target_ID)

        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)

        return x,y
        


# if __name__=="__main__":
#     f = os.path.join(os.path.dirname(os.getcwd()),'dataset/training_data.npy')
#     print(f)
    
#     training_dataset = Dataset(f)

#     training_dataloader = data.DataLoader(dataset=training_dataset,
#                                         batch_size=2,
#                                         shuffle=True)
#     x, y = next(iter(training_dataloader))

#     print(f'x = shape: {x.shape}; type: {x.dtype}')
#     print(f'x = min: {x.min()}; max: {x.max()}')
#     print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')