''' 
data_generator.py: This file creates custom datasets for our model.
'''
__author__ = "A.Utkarsh(ayushutkarsh@gmail.com) and M.Gupta(mgmayank18@gmail.com)"


from torch.utils.data import Dataset
import os

class FingerPrintDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.samples = []
        self.transform =transform
        self.root_dir = root_dir
        self.__init__dataset()
        self.label_dict={}
    def __init__dataset(self):
        counter=1
        for folders in os.listdir(self.root_dir):
            folder_path=os.path.join(self.root_dir,folders)
            for image_name in os.listdir(folders):
                img_path = os.path.join(folder_path,image_name)
                filename = image_name.split("/")[-1]
                foldername = folder_path
                tags = filename.split("_")
                if len(tags) == 5:
                    key = foldername+"/"+"_".join(tags[0:4])    
                elif len(tags) == 3:
                    key = foldername+"/"+tags[0]+"_"+tags[2]
                elif len(tags) == 4:
                    key = foldername+"/"+tags[0]+"_sess_"+tags[2]+"_"+tags[3]
                elif len(tags) == 2:
                    key = foldername+tags[0][1:]
                if key not in self.label_dict:
                    self.label_dict[key]=counter
                    counter +=1
                self.samples.append((img_path,self.label_dict[key]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.samples[idx][0]
        label = self.samples[idx][1]
        image = io.imread(img_path)
        sample = {'image': image, 'label': label}
        return sample
