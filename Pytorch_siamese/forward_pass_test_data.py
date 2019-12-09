#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
from networks import EmbeddingNet, TripletNet
from data_generator import FingerPrintDataset, FingerPrintDataset_rgb
from losses import TripletLoss
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit,final_test_epoch


# In[2]:


#For testing forward pass, we shall be using the CPU. Hence, cuda = False
cuda =False
# Set up the network and training parameters
import torchvision
from torch import nn
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric


# In[3]:


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# In[4]:


model = EmbeddingNet()
model.load_state_dict(torch.load('./saved_model/titi'))
model.eval()


# In[5]:


final_test_epoch('/Users/ayush/projects/my_pytorch/probe' ,'/Users/ayush/projects/my_pytorch/gallery','/Users/ayush/projects/my_pytorch/fp_output_txt',model,metrics=[AverageNonzeroTripletsMetric()],transform=transforms.Compose([
                                 transforms.ToTensor()]))


# In[7]:


get_ipython().system('rm /Users/ayush/projects/my_pytorch/probe/.DS_Store')


# In[ ]:




