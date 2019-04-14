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
count=500
margin=0.5

### TRIPLET LOSS ###

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
        out = tf.layers.batch_normalization(fc1, training=True,name='batchnorm')
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
        out = tf.layers.batch_normalization(fc4, training=True,name='text_batchnorm')
        tf.summary.histogram('Text_batch_norm_layer', out)
        out = tf.identity(out, name='my_text_embedding')
        tf.summary.histogram('Text_final_activation', out)
    return out
    
###############################################################################################################
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
#text_embeddings = text_net(text_dict, reuse, is_training)
image_embedding_mean_norm = tf.reduce_mean(tf.norm(image_embeddings, axis=1))
tf.summary.scalar("embedding_image_mean_norm", image_embedding_mean_norm)
#text_embedding_mean_norm = tf.reduce_mean(tf.norm(text_embeddings, axis=1))
#tf.summary.scalar("embedding_text_mean_norm", text_embedding_mean_norm)
# update operation to update batch-norm variables
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
#    my_loss, num_tii, num_itt = batch_all_triplet_loss(labels, image_embeddings, text_embeddings, margin, squared=False)
#    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op')
#    minimizer = optimizer.minimize(my_loss)
    
    
##########################################################################################################################
with tf.name_scope("Init"):
    merged = tf.summary.merge_all()
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
        saver = tf.train.import_meta_graph('models/run_6/model'+str(count)+'.ckpt.meta')  # Put the meta file here
        saver.restore(sess, "models/run_6/model"+str(count)+".ckpt")
#        saver.restore(sess, "models/run_6/checkpoint")
        print("Model Restored")
        print("Init Done")
        test_writer = tf.summary.FileWriter('./Graph_run6/test',sess.graph)
        image_embeddings = graph.get_tensor_by_name("ConvNet/my_feature_embedding:0")
        image_dict = graph.get_tensor_by_name("inputs/Placeholder:0") #same as above
        
        # Running for the Total_size/batch_size times
        count=0
        for latent_images in test_list:
            expanded_image = np.float32(np.expand_dims(latent_images,0))
            feed_dict_batch= {image_dict: expanded_image,is_training:False}
            test_embedding = sess.run(image_embeddings, feed_dict=feed_dict_batch)
            test_embeddings.append(test_embedding)
            summary = sess.run([merged], feed_dict=feed_dict_batch)
            #test_writer.add_summary(summary, count)
            count+=1
            if count%250==0:
                print(count,"test images done")
            if count==10:
                break
#            print(test_embeddings.shape)
            print(test_embedding[0:10,0:10])
        count=0
        for latent_images in template_list:
            expanded_image = np.float32(np.expand_dims(latent_images,0))
            feed_dict_batch = {image_dict: expanded_image,is_training:False}
            template_embedding = sess.run(image_embeddings, feed_dict = feed_dict_batch)
            template_embeddings.append(template_embedding)
            summary = sess.run([merged], feed_dict=feed_dict_batch)
            #test_writer.add_summary(summary, count)
            count+=1
            if count%250==0:
                print(count,"template images done")
            if count==10:
                break
    print('Signature Extraction Done.')
    sys.stdout.flush()

    hf = h5py.File('/work/cvma/FP/data/fp_sig_data.h5', 'w')
    hf.create_dataset('test_embeddings', data=test_embeddings)
    hf.create_dataset('template_embeddings', data=template_embeddings)
    hf.close()
    np.savetxt('/work/cvma/FP/data/test_embeddings.txt', test_embeddings,delimiter=',',newline='\n')
    np.savetxt('/work/cvma/FP/data/template_embeddings.txt', template_embeddings,delimiter=',',newline='\n')

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
        
