import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class SignMNISTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        #self.imgData = np.array(pd.read_csv(csv_file), dtype='float32')
        self.mnistData = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        #return self.imgData.shape[0]
        return len(self.mnistData)
    
    def __getitem__(self, idx):
        #sample = {'image': imgData[idx, 1:].reshape((28, 28)), 'label': imgData[idx, 0]}
        currImg = np.array(self.mnistData.iloc[idx], dtype='float32')
        currImg = currImg[1:].reshape((28, 28))
        #currImg = np.array([[[(currImg[x, y])] for y in range(currImg.shape[1])] for x in range(currImg.shape[0])])
        #currImg = currImg[1:]
        label = torch.tensor(int(self.mnistData.iloc[idx, 0]))
        
        if self.transform:
            currImg = self.transform(currImg)
            
        return (currImg, label)