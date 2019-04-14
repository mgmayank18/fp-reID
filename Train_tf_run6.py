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

batch_size = 100
Alpha = 0.5
margin = Alpha

# Generators
training_generator = data_generator('/work/cvma/FP/data/Train_Filelist.txt',batch_size,True).generate()

#No need for positive mask in Multimodal Triplet. Anchor and Positives will be fed through data inputs.'''
def _pairwise_distances(image_embeddings, text_embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product11 = tf.matmul(image_embeddings, tf.transpose(image_embeddings))
    dot_product22 = tf.matmul(text_embeddings, tf.transpose(text_embeddings))
    dot_product12 = tf.matmul(image_embeddings, tf.transpose(text_embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm11 = tf.diag_part(dot_product11)
    square_norm22 = tf.diag_part(dot_product22)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm11, 0) - 2.0 * dot_product12 + tf.expand_dims(square_norm22, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

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
    
    return triplet_loss, num_positive_triplets_tii, num_positive_triplets_itt

#############################################################################################################

#############################################################################################################

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
        flat = tf.contrib.layers.flatten(conv3)
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
    reuse = False
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
    init = tf.global_variables_initializer()
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    
##########################################################################################################################
################################################# TRAINING SESSION ######################################################

with tf.Session() as sess:
    print("Initializing Session")
    sess.run(init)
    print("Init Done")
    train_writer = tf.summary.FileWriter('./Graph_run_6/train',sess.graph)
    test_writer = tf.summary.FileWriter('./Graph_run_6/test',sess.graph)
    global_step = 0
    count = 1
    for [images, texts], labels_input in training_generator:
        if count < 10:
            print("Running Batch Number", count)
        feed_dict_batch = {image_dict: images, text_dict: texts, labels: labels_input,is_training: True}
        loss_val, _num_tii, _num_itt, minimized = sess.run([my_loss, num_tii, num_itt, minimizer], feed_dict=feed_dict_batch)
        if count%20 == 0 or count < 10:
            print(count, " -- Loss Val: ", loss_val, " -- Trip_TII: ", _num_tii, " -- Trip_ITT: ", _num_itt)
            summary, _ = sess.run([merged,minimizer], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary, count)
        if count%500 == 0:
            save_path = saver.save(sess, "models/run_6/model"+str(count)+".ckpt")
            print("Model saved in path: %s" % save_path)
        if count%100000 == 0:
            break
        count+=1
