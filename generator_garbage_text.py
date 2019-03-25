import os
import random
import numpy as np
import cv2

class data_generator(object):
    'Generates data for Keras'
    def __init__(self, datapath_file, batch_size = 32, shuffle = True):
        print("Initializing Generator...")
        'Initialization'
        if shuffle == False:
            raise NotImplementedError("Shuffle = False is not yet supported")

        self.datapath_file = datapath_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        image_paths = []
        text_paths = []
        labels = []
        with open(datapath_file, 'r') as datafile:
            for line in datafile:
                [image_path, text_path, label] = line.split(" ")
                image_paths.append(image_path)
                text_paths.append(text_path)
                labels.append(label)
        self.labels=labels
        self.image_paths = image_paths
        self.text_paths = text_paths
        self.indices = np.arange(len(image_paths))
        np.random.shuffle(self.indices)
        self.counter = 332000
        self.len_data = len(image_paths)
        
   #Minutae Data Preprocessing

    def on_epoch_end(self):
        indices = self.indices
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)
        self.counter = 0
            
            
    def generate(self):
        
        'Generates batches of samples' 
        # Infinite loop
        while True:
            # Generate ImageIDs
            if self.counter + self.batch_size >= self.len_data:
                np.random.shuffle(self.indices)
                self.counter = 0
            data_indices = self.indices[self.counter:self.counter+self.batch_size]
            image_list = [self.image_paths[x] for x in data_indices]
            text_list = [self.text_paths[x] for x in data_indices]
            L = [self.labels[x] for x in data_indices]
            I, T = self.__load_data(image_list, text_list)
            self.counter+=self.batch_size
            yield [I, T], L

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __load_data(self, image_list, text_list):
        def crop_top(df,crop_size):
            #this will take top n inputs for df
            if df.shape[0]<crop_size:
                d_zero = np.zeros((crop_size,df.shape[1]), dtype=int)
                d_zero[:df.shape[0], :df.shape[1]]=df
                return d_zero
            else:
                return df[0:crop_size,:]

        def is_file_empty(filename):
            return os.stat(filename).st_size == 0

        def preprocess_xyt_file(filename, crop_size=50):
            df=np.loadtxt(filename, delimiter=' ')
            if len(np.shape(df)) != 1:
                df = df[df[:,3].argsort()[::-1]]
            else:
                df = np.expand_dims(df,0)
            normalization_factor = np.array([512.0,512.0,360.0,100.0])
            df_new = crop_top(df,crop_size)/normalization_factor
            return df_new.flatten()

        'Loads data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        image_data = []
        text_data = []
        for image in image_list:
            image_data.append(cv2.imread(image))
        ####################################################################
                  # garbage text #
        ####################################################################
        my_len = len(image_list)
        pixels_text=np.random.randint(512, size=(my_len, 50,2))
        degree_text=np.random.randint(360, size=(my_len, 50,1))
        confidence_text=np.random.rand(my_len, 50,1)*100
        text_data=np.concatenate((pixels_text,degree_text,confidence_text) ,axis=2)
        #for text in text_list:
         #   text_data.append(preprocess_xyt_file(text))
        return np.stack(image_data), np.stack(text_data)
