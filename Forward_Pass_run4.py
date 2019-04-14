#Import libs
import time
import os, os.path
import random
import cv2
import matplotlib
import functools
import matplotlib.pyplot as plt
import tensorflow as tf
from Data_Generator import data_generator
import glob
import pandas as pd
import numpy as np
import faiss
import h5py
import sys
import time

#Hyperparameters

load_image_data_flag = 1
load_sig_data=0
batch_size = 1

###################################### tensorflow network ###################################################

def image_net(image_dict, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        image_input= image_dict
        image_input = tf.reshape(image_input, shape=[-1,512,512,3])
        conv1 = tf.layers.conv2d(image_input, 32, 3, activation=tf.nn.relu)
        conv1 = tf.layers.average_pooling2d(conv1,2,4)
        conv2 = tf.layers.conv2d(conv1, 128, 3, activation = tf.nn.sigmoid)
        conv2 = tf. layers.average_pooling2d(conv2,2,4)
        conv3 = tf.layers.conv2d(conv2, 512, 3, activation = tf.nn.sigmoid)
        conv3 = tf.layers.average_pooling2d(conv3,2,4)
        flat = tf.contrib.layers.flatten(conv3,name='flat')
        tf.summary.histogram('Image_flatten', flat)
        fc1 = tf.layers.dense(flat, 512, activation=None, name='fc1_dense')
        out = tf.layers.batch_normalization(fc1, training=is_training,name='batchnorm')
        tf.summary.histogram('Image_batch_norm_layer', out)
        out = tf.identity(out, name ='my_feature_embedding')
        tf.summary.histogram('Image_final_activation', out)
        
    return out 

def text_net (text_dict, reuse, is_training): 
    with tf.variable_scope('TextNet', reuse=reuse):
        text_input = text_dict
        fc1 = tf.layers.dense(text_input, 300, activation = tf.nn.sigmoid)
        fc2 = tf.layers.dense(fc1, 400, activation = tf.nn.sigmoid)
        fc3 = tf.layers.dense(fc2, 450, activation= tf.nn.sigmoid)
        fc4 = tf.layers.dense(fc3, 512, activation = None)
        tf.summary.histogram('Text_dense_4', fc4)
        out = tf.layers.batch_normalization(fc4, training=is_training,name='text_batchnorm')
        tf.summary.histogram('Text_batch_norm_layer', out)
        out = tf.identity(out, name='my_text_embedding')
        tf.summary.histogram('Text_final_activation', out)
    return out
    
###################################################################################################################
tf.reset_default_graph()
with tf.name_scope("inputs"):
    reuse = True
    learning_rate = 0.001
    image_dict = tf.placeholder(tf.float32, shape=(batch_size,512,512,3))
    text_dict = tf.placeholder(tf.float32, shape = (batch_size,200))
    labels = tf.placeholder(tf.float32, shape=(batch_size))
    is_training = tf.placeholder_with_default(False, shape=[], name='training')

####################################################################################################################

image_embeddings = image_net(image_dict, reuse, is_training)
text_embeddings = text_net(text_dict, reuse, is_training)
image_embedding_mean_norm = tf.reduce_mean(tf.norm(image_embeddings, axis=1))
tf.summary.scalar("embedding_image_mean_norm", image_embedding_mean_norm)
text_embedding_mean_norm = tf.reduce_mean(tf.norm(text_embeddings, axis=1))
tf.summary.scalar("embedding_text_mean_norm", text_embedding_mean_norm)
# update operation to update batch-norm variables
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    my_loss, num_tii, num_itt = batch_all_triplet_loss(labels, image_embeddings, text_embeddings, margin, squared=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op')
    minimizer = optimizer.minimize(my_loss)
    
    
##########################################################################################################################
with tf.name_scope("Init"):
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('./Graph_run4/test',sess.graph)
    init = tf.global_variables_initializer()
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
###########################################################################################################################
print("Reading Images")
val_files_path="/work/cvma/FP/data/Val_Filelist.txt"

template_list = []
template_labels = []
test_list = []
test_labels = []
test_embeddings=[]
template_embeddings=[]

if load_image_data_flag==0:
    count = 0
    with open(val_files_path, 'r') as val_files:
        for line in val_files:
            if count%100 == 0:
                print(count, "lines done.")
            [image_path, minutae_path, label] = line.split(" ")
            image_data = cv2.imread(image_path)
            if image_path.split("_")[-1][0:7] == "blurred":
                test_list.append(image_data)
                test_labels.append(label.strip())
            else:
                template_list.append(image_data)
                template_labels.append(label.strip())
            #if count==200:
            #    break
            count+=1
    template_list=np.array(template_list)
    template_labels=np.array(template_labels,dtype=int)
    test_list=np.array(test_list)
    test_labels=np.array(test_labels,dtype=int)
    hf = h5py.File('/work/cvma/FP/data/fp_data_read.h5', 'w')
    hf.create_dataset('template_list', data=template_list)
    hf.create_dataset('template_labels', data=template_labels)
    hf.create_dataset('test_list', data=test_list)
    hf.create_dataset('test_labels', data=test_labels)

    hf.close()
else:
    hf = h5py.File('/work/cvma/FP/data/fp_data_read.h5', 'r')
    template_list = hf['template_list'][:]
    template_labels = hf['template_labels'][:]
    test_list = hf['test_list'][:]
    test_labels = hf['test_labels'][:]
    hf.close()
    print("Template and Test List/Labels shapes: ",template_list.shape,template_labels.shape,test_list.shape,test_labels.shape)
    template_list=np.array(template_list)
    template_labels=np.array(template_labels)
    test_list=np.array(test_list)
    test_labels=np.array(test_labels)
    print("Data Loaded")
#############################################################################################################################
if load_sig_data==0:    
    with tf.Session() as sess:
        print("Initializing Session")
        sess.run(init)
        saver = tf.train.import_meta_graph('models/run_4/model"+str(count)+".ckpt.meta')  # Put the meta file here
        saver.restore(sess, "models/run_4/model______.ckpt")
        print("Model Restored")
        print("Init Done")
        image_embeddings = graph.get_tensor_by_name("ConvNet/my_feature_embedding:0")
        image_dict = graph.get_tensor_by_name("image_dict:0") #same as above
        
        # Running for the Total_size/batch_size times
        count=0
        for latent_images in test_list:
            print(latent_images.shape)
            expanded_image = np.float32(np.expand_dims(latent_images,0))
            feed_dict_batch= {image_dict: expanded_image,is_training:False}
            test_embedding = sess.run(image_embeddings, feed_dict=feed_dict_batch)
            test_embeddings.append(test_embedding)
            summary, _ = sess.run([merged,minimizer], feed_dict=feed_dict_batch)
            test_writer.add_summary(summary, count)
            count+=1
            if count%250==0:
                print(count,"test images done")
        count=0
        for latent_images in template_list:
            expanded_image = np.float32(np.expand_dims(latent_images,0))
            feed_dict_batch = {image_dict: expanded_image,is_training:False}
            template_embedding = sess.run(image_embeddings, feed_dict = feed_dict_batch)
            template_embeddings.append(template_embedding)
            summary, _ = sess.run([merged,minimizer], feed_dict=feed_dict_batch)
            test_writer.add_summary(summary, count)
            count+=1
            if count%250==0:
                print(count,"template images done")
    print('Signature Extraction Done.')
    sys.stdout.flush()

    hf = h5py.File('/work/cvma/FP/data/fp_sig_data.h5', 'w')
    hf.create_dataset('test_embeddings', data=test_embeddings)
    hf.create_dataset('template_embeddings', data=template_embeddings)
    hf.close()
else:
    hf = h5py.File('/work/cvma/FP/data/fp_sig_data.h5', 'r')
    test_embeddings = hf['test_embeddings'][:]
    template_embeddings = hf['template_embeddings'][:]
    hf.close()
    print("Test and Template Embeddings Shapes: ",test_embeddings.shape,template_embeddings.shape)
    test_embeddings=np.squeeze(np.array(test_embeddings),1)
    template_embeddings=np.squeeze(np.array(template_embeddings),1)
    print("Sigs Read")
    np.savetxt('/work/cvma/FP/data/test_embeddings.txt', test_embeddings,delimiter=',',newline='\n')
    np.savetxt('/work/cvma/FP/data/template_embeddings.txt', template_embeddings,delimiter=',',newline='\n')
    
print(template_embeddings.shape)
print(test_embeddings.shape)

d = 512
nb = template_embeddings.shape[0]
nq = test_embeddings.shape[0]

index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
sys.stdout.flush()
index.add(template_embeddings)                  # add vectors to the index
print(index.ntotal)
sys.stdout.flush()
k = 1
print("Doing Sanity Check")
sys.stdout.flush()
D, I = index.search(template_embeddings, k) # sanity check
print(I)
print(D)

print("Sanity Check Over")
sys.stdout.flush()
D, I = index.search(test_embeddings, k)     # actual search
print(I[10:15])                   # neighbors of the 5 first queries
print(I[-15:])                  # neighbors of the 5 last queries
sys.stdout.flush()

np.savetxt("/work/cvma/FP/faiss_output.txt",I,delimiter=",",fmt='%i')
        
