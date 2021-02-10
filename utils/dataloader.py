import torch
from torch.utils import data
from cv2 import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

        x = np.array(cv2.split(cv2.imread(input_ID)))
        y = cv2.imread(target_ID, cv2.IMREAD_GRAYSCALE)
        plt.imshow(cv2.imread(input_ID))
        plt.show()

        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)

        return x,y
        
if __name__=="__main__":
    inputs = ['dataset/inputs/ref_landcovernet_v1_labels_28QDE_09_0.tif', 
            'dataset/inputs/ref_landcovernet_v1_labels_28QDE_09_1.tif']
    targets = ['dataset/target/ref_landcovernet_v1_labels_28QDE_09.tif',
            'dataset/target/ref_landcovernet_v1_labels_28QDE_09.tif']

    training_dataset = Dataset(inputs=inputs,targets=targets)

    training_dataloader = data.DataLoader(dataset=training_dataset,
                                        batch_size=2,
                                        shuffle=True)
    x, y = next(iter(training_dataloader))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')