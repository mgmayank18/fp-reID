#Import libs
import time
import os, os.path
import random
import cv2
import keras
import matplotlib
import functools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import optimizers
import keras.backend as K
import glob
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#Read Data Paths

images = sorted(glob.glob('/work/cvma/FP/data/*/*[!_xyt]/*'))
xyt = sorted(glob.glob('/work/cvma/FP/data/*/*_xyt/*'))
image_xyt_pairs = [list(a) for a in zip(images,xyt)]

#Image Data Preprocessing

label_dict = {}
label_counter = 0
label = []
image_list = []
text_list = []

def assign_label(key):
    global label_counter
    if key in label_dict.keys():
        label.append(label_dict[key])
        return 0
    else:
        label_dict[key] = label_counter
        label.append(label_counter)
        label_counter=label_counter+1
        #Do Something
        return 0
for image in images:
    filename = image.split("/")[-1]
    foldername = "/".join(image.split("/")[0:-1])
    tags = filename.split("_")
    if len(tags) == 5:
        key = foldername+"/"+"_".join(tags[0:4])
        assign_label(key)
        

    elif len(tags) == 3:
        key = foldername+"/"+tags[0]+"_"+tags[2]
        assign_label(key)
    #elif len(tags) == 4:
        
    #elif len(tags) == 2:
    
#Minutae Data Preprocessing
def crop_top(df,crop_size):
    #this will take top n inputs for df
    if df.shape[0]<crop_size:
        d_zero = np.zeros((crop_size,df.shape[1]), dtype=int)
        d_zero[:df.shape[0], :df.shape[1]]=df
        return d_zero
    else:
        return df[0:crop_size,:]
    
def preprocess_xyt_file(filename, crop_size):
    df=pd.read_csv(filename, sep=' ',header=None)
    df = np.array(df)
    np.sort(df, axis=0)
    df = df[df[:,3].argsort()[::-1]]

    normalization_factor = np.array([512,512,360,100])
    df_new = crop_top(df,crop_size)/normalization_factor
    
    return df_new.flatten()



"""Define functions to create the triplet loss with online triplet mining."""

ses = tf.Session()

image_embeddings = tf.random_uniform([1000,512], minval=0, maxval=1, dtype=tf.float32)
image_embeddings = tf.constant([[1.0,1.0,1.0],[3.0,3.0,3.0]])
text_embeddings = tf.random_uniform([1000,512], minval=0, maxval=1, dtype=tf.float32)
text_embeddings = tf.constant([[1.0,2.0,1.0],[1.0,2.0,3.0]])
y_pred = tf.concat([image_embeddings, text_embeddings], 0)

y1,y2 =tf.split(y_pred, num_or_size_splits=2, axis=0)
tf.shape(image_embeddings)

def euclid(image,text):
    return euclidean_distances(image,text,squared=True)

def _pairwise_distances(image_embeddings, text_embeddings, squared=True):
    #Returns Pairwise Euclidean Distances
    if squared == True:
        distances = tf.py_func(euclid, [image_embeddings, text_embeddings], tf.float32)
    else:
        distances = tf.py_func(euclidean_distances,[image_embeddings, text_embeddings], tf.float32)        
    return distances

#No need for positive mask in Multimodal Triplet. Anchor and Positives will be fed through data inputs.

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i == j, k is distinct
        - i == j and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i == j and k is distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_equal_j = tf.expand_dims(indices_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    valid_indices = tf.logical_and(tf.logical_and(i_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(valid_indices, valid_labels)

    return mask

#No need for triplet mask. Negative Mask, serves as triplet mask for both L_tii and L_itt.

def batch_all_triplet_loss(labels, image_embeddings, text_embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    
    # Get the pairwise distance matrix for L_itt
    pairwise_dist = _pairwise_distances(image_embeddings, text_embeddings)
    
    # Get the pairwise distance matrix for L_tii
    pairwise_dist_transpose = tf.transpose(pairwise_dist)

    # shape (batch_size, batch_size, 1)
    #Anchor Positive Dist is same for L_tii and L_itt
    #identity_mask = tf.eye(tf.shape(pairwise_dist))
    #anchor_positive_dist = tf.expand_dims(tf.multiply(identity_mask,pairwise_dist), 2)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    print(tf.shape(anchor_positive_dist))
    #assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1) #L_itt
    anchor_negative_dist_transpose = tf.expand_dims(pairwise_dist_transpose, 1) #L_tii
    #assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    #assert anchor_negative_dist_transpose.shape[1] == 1, "{}".format(anchor_negative_dist_transpose.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    L_itt = anchor_positive_dist - anchor_negative_dist + margin
    L_tii = anchor_positive_dist - anchor_negative_dist_transpose + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    L_itt = tf.multiply(mask, L_itt)
    L_tii = tf.multiply(mask, L_tii)

    # Remove negative losses (i.e. the easy triplets)
    L_itt = tf.maximum(L_itt, 0.0)
    L_tii = tf.maximum(L_tii, 0.0)
    

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets_tii = tf.to_float(tf.greater(L_tii, 1e-16))
    valid_triplets_itt = tf.to_float(tf.greater(L_itt, 1e-16))
    
    num_positive_triplets_tii = tf.reduce_sum(valid_triplets_tii)
    num_positive_triplets_itt = tf.reduce_sum(valid_triplets_itt)
    
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = (num_positive_triplets_tii + num_positive_triplets_itt) / (2*num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = (tf.reduce_sum(L_tii) + tf.reduce_sum(L_itt)) / (num_positive_triplets_tii + num_positive_triplets_itt + 1e-16)
    
    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


image_branch=Sequential()
resnet50_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(512,512,3))
image_branch.add(resnet50_model)
#model.add(keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
#model.add(Flatten())
image_branch.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'))
image_branch.add(Dense(1024,kernel_initializer=glorot_normal(seed=None), activation='sigmoid'))
image_branch.add(Dense(512, kernel_initializer=glorot_normal(seed=None), activation=None))
image_branch.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

text_branch = Sequential()
text_branch.add(Dense(400, kernel_initializer=glorot_normal(seed=None), activation='sigmoid', input_shape = (200,)))
text_branch.add(Dense(512, kernel_initializer=glorot_normal(seed=None), activation = None))
text_branch.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

#model_combined = Sequential()
mergedOut = Concatenate()([image_branch.output,text_branch.output])
model_combined=Model(inputs=[image_branch.input, text_branch.input], outputs=mergedOut)

labels=[0,1]
y_true = labels
embeddings = tf.concat([image_embeddings, text_embeddings], 0)

#def triplet_loss(image_embeddings, text_embeddings, margin, squared=True):
def _batch_all_triplet_loss(labels,image_embeddings, text_embeddings, margin):
    return batch_all_triplet_loss(labels,image_embeddings, text_embeddings, margin, squared=True)

def triplet_loss(margin):
    @functools.wraps(_batch_all_triplet_loss)
    def loss(labels, embeddings):
        image_embeddings, text_embeddings =tf.split(embeddings, num_or_size_splits=2, axis=0)
        return _batch_all_triplet_loss(labels,image_embeddings, text_embeddings, margin)
    return loss

triplet_loss = triplet_loss(0.5)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss=triplet_loss, optimizer=sgd)