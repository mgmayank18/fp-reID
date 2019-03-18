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

#Hyperparameters

batch_size = 1

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
        flat = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(flat, 512, activation=tf.nn.relu)
        out = tf.layers.batch_normalization(fc1, training=is_training)
        
    return out 

# The text_net consists of 4 fc layers of outputs 400, 512, 1024, 512 respectively

def text_net (text_dict, reuse, is_training):
    
    with tf.variable_scope('model', reuse=reuse):
        
        text_input = text_dict
        fc1 = tf.layers.dense(text_input, 300, activation = tf.nn.sigmoid)
        fc2 = tf.layers.dense(fc1, 400, activation = tf.nn.sigmoid)
        fc3 = tf.layers.dense(fc2, 450, activation= tf.nn.sigmoid)
        fc4 = tf.layers.dense(fc3, 512, activation = tf.nn.sigmoid)
        out = tf.layers.batch_normalization(fc4, training=is_training)
    
    return out
    
#### my chindi network
reuse = None
learning_rate = 0.001
batch_size=1
image_dict = tf.placeholder(tf.float32, shape=(batch_size,512,512,3))
text_dict = tf.placeholder(tf.float32, shape = (batch_size,200))
labels = tf.placeholder(tf.float32, shape=(batch_size))


image_embeddings = image_net(image_dict, reuse, True)
text_embeddings = text_net(text_dict, reuse, True)
image_embedding_mean_norm = tf.reduce_mean(tf.norm(image_embeddings, axis=1))
tf.summary.scalar("embedding_image_mean_norm", image_embedding_mean_norm)
text_embedding_mean_norm = tf.reduce_mean(tf.norm(text_embeddings, axis=1))
tf.summary.scalar("embedding_text_mean_norm", text_embedding_mean_norm)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
batch_size = 1
image_dict = tf.placeholder(tf.float32, shape=(batch_size,512,512,3))
labels = tf.placeholder(tf.float32, shape=(batch_size))

print("Reading Images")
val_files_path="/work/cvma/FP/data/Val_Filelist.txt"

template_list = []
template_labels = []
test_list = []
test_labels = []

count = 0
with open(val_files_path, 'r') as val_files:
    for line in val_files:
        if count%100 == 0:
            print(count, "lines done.")
        [image_path, minutae_path, label] = line.split(" ")
        image_data = cv2.imread(image_path)
        if image_path.split("_")[-1][0:7] == "blurred":
            test_list.append(image_data)
            test_labels.append(label)
        else:
            template_list.append(image_data)
            template_labels.append(label)
        if count==5000:
            break
        count+=1
            
test_embeddings=[]
template_embeddings=[]

with tf.Session() as sess:
    print("Initializing Session")
    sess.run(init)
    saver.restore(sess, "models/run_3/model100000.ckpt")
    print("Model Restored")
    print("Init Done")
    # Running for the Total_size/batch_size times
    for latent_images in test_list:
        expanded_image = np.expand_dims(latent_images,0)
        print(expanded_image.shape)
        feed_dict_batch = {image_dict: expanded_image}
        test_embedding = sess.run(image_embeddings, feed_dict=feed_dict_batch)
        test_embeddings.append(test_embedding)
    for latent_images in template_list:
        feed_dict_batch = {image_dict: np.expand_dims(latent_images,0)}
        template_embedding = sess.run(image_embeddings, feed_dict = feed_dict_batch)
        template_embeddings.append(template_embedding)