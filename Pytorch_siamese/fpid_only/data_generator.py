#!/usr/bin/env python
# coding: utf-8

# In[16]:





# In[18]:


from torch.utils.data import Dataset
import os
import torch
import skimage.io as io
import warnings
from PIL import Image

class FingerPrintDataset(Dataset):
    def __init__(self,root_dir,train=True,transform=None):
        self.samples = []
        self.train = train
        self.transform =transform
        self.root_dir = root_dir
        self.label_dict={}
        self.data=[]
        self.targets=[]
        self.__init__dataset()
        
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
        
    def __init__dataset(self):
        counter=1
        if self.train:
            datadir=self.root_dir+"train"
        else:
            datadir=self.root_dir+"test"
        print(datadir)
        for image_name in os.listdir(datadir):
            img_path = os.path.join(datadir,image_name)
            filename = image_name.split("/")[-1]
            #foldername = folder_path
            tags = filename.split("_")
            if len(tags) == 3:
                key = tags[0]+"_"+tags[2]
            elif len(tags) == 4:
                key = tags[0]+"_"+tags[2]+"_"+tags[3]
            elif len(tags) ==2:
                key = tags[0]
            if key not in self.label_dict:
                self.label_dict[key]=counter
                counter +=1
            self.samples.append((img_path,self.label_dict[key])) 
            image = Image.open(img_path)
            image = image.convert('L')
            
            if self.transform is not None:
                image = self.transform(image)
            self.data.append(image)
            self.targets.append(self.label_dict[key])
        self.targets=torch.Tensor(self.targets)  
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx],int(self.targets[idx])
    

    
class FingerPrintDataset_rgb(Dataset):
    def __init__(self,root_dir,train=True,transform=None):
        self.samples = []
        self.train = train
        self.transform =transform
        self.root_dir = root_dir
        self.label_dict={}
        self.data=[]
        self.targets=[]
        self.__init__dataset()
        
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
        
    def __init__dataset(self):
        counter=1
        if self.train:
            datadir=self.root_dir+"train"
        else:
            datadir=self.root_dir+"test"
        print(datadir)
        for image_name in os.listdir(datadir):
            img_path = os.path.join(datadir,image_name)
            filename = image_name.split("/")[-1]
            #foldername = folder_path
            tags = filename.split("_")
            if len(tags) == 3:
                key = tags[0]+"_"+tags[2]
            elif len(tags) == 4:
                key = tags[0]+"_"+tags[2]+"_"+tags[3]
            elif len(tags) ==2:
                key = tags[0]
            if key not in self.label_dict:
                self.label_dict[key]=counter
                counter +=1
            self.samples.append((img_path,self.label_dict[key])) 
            image = Image.open(img_path)
            image = image.convert('RGB')
            
            if self.transform is not None:
                image = self.transform(image)
            self.data.append(image)
            self.targets.append(self.label_dict[key])
        self.targets=torch.Tensor(self.targets)  
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx],int(self.targets[idx])
    

# In[14]:





# In[15]:





# In[ ]:




