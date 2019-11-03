#Import libs
import time
import os, os.path
import random
import cv2
import keras
import matplotlib
import functools
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model
from keras import optimizers
import keras.backend as K
import glob
import numpy as np

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#Hyperparameters

BatchSize = 64
Alpha = 0.5

#Read Data Paths

images = sorted(glob.glob('/local/manasa/FP/data/*/*[!_xyt]/*'))
xyt = sorted(glob.glob('/local/manasa/FP/data/*/*_xyt/*'))
image_xyt_pairs = [list(a) for a in zip(images,xyt)]

#Image Data Preprocessing

label_dict = {}
label_counter = 0
labels = []
image_list = []
text_list = []

#Minutae Data Preprocessing
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

def assign_label(key):
    global label_counter
    if key in label_dict.keys():
        labels.append(label_dict[key])
        return 0
    else:
        label_dict[key] = label_counter
        labels.append(label_counter)
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
        image_data = cv2.imread(image)
        image_list.append(image_data)
        Index = images.index(image)
        text_list.append(preprocess_xyt_file(xyt[Index]))

    elif len(tags) == 3:
        key = foldername+"/"+tags[0]+"_"+tags[2]
        assign_label(key)
        image_data = cv2.imread(image)
        image_list.append(image_data)
        Index = images.index(image)
        text_list.append(preprocess_xyt_file(xyt[Index]))
    elif len(tags) == 4:
        Index = images.index(image)
        if is_file_empty(xyt[Index]):
            continue
        key = foldername+"/"+tags[0]+"_sess_"+tags[2]+"_"+tags[3]
            assign_label(key)
        image_list.append(image_data)
            text_list.append(preprocess_xyt_file(xyt[Index]))
    elif len(tags) == 2:
        key = foldername+tags[0][1:]
        assign_label(key)
        image_list.append(image_data)
        Index = images.index(image)
        text_list.append(preprocess_xyt_file(xyt[Index])) 
        
print("Image List: ", len(image_list), "Text List : ", len(text_list), "Labels : ", len(labels))

"""Define functions to create the triplet loss with online triplet mining."""

ses = tf.Session()

image_embeddings = tf.random_uniform([1000,512], minval=0, maxval=1, dtype=tf.float32)
image_embeddings = tf.constant([[1.0,1.0,1.0],[3.0,3.0,3.0]])
text_embeddings = tf.random_uniform([1000,512], minval=0, maxval=1, dtype=tf.float32)
text_embeddings = tf.constant([[1.0,2.0,1.0],[1.0,2.0,3.0]])
y_pred = tf.concat([image_embeddings, text_embeddings], 0)

y1,y2 =tf.split(y_pred, num_or_size_splits=2, axis=0)
tf.shape(image_embeddings)

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
    
    return triplet_loss


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

input_images = Input(shape=(512,512,3))
resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(512,512,3))
x = resnet50_model(input_images)
x = GlobalAveragePooling2D(data_format='channels_last')(x)
x = Dense(1024,kernel_initializer=glorot_normal(seed=None), activation='sigmoid')(x)
x = Dense(512, kernel_initializer=glorot_normal(seed=None), activation=None)(x)
output_image = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
#x.summary()

input_text = Input(shape=(200,))
y = Dense(400, kernel_initializer=glorot_normal(seed=None), activation='sigmoid', input_shape = (200,))(input_text)
y = Dense(512, kernel_initializer=glorot_normal(seed=None), activation = None)(y)
output_text = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(y)
#y.summary()

#model_combined = Sequential()
mergedOut = Concatenate(axis=0)([output_image,output_text])
model_combined=Model(inputs=[input_images, input_text], outputs=mergedOut)


#y_true = labels
#embeddings = tf.concat([image_embeddings, text_embeddings], 0)

#def triplet_loss(image_embeddings, text_embeddings, margin, squared=True):
def _batch_all_triplet_loss(labels,image_embeddings, text_embeddings, margin):
    return batch_all_triplet_loss(labels,image_embeddings, text_embeddings, margin, squared=True)

def triplet_loss(margin):
    @functools.wraps(_batch_all_triplet_loss)
    def loss(labels, embeddings):
        image_embeddings, text_embeddings =tf.split(embeddings, num_or_size_splits=2, axis=0)
        return _batch_all_triplet_loss(labels,image_embeddings, text_embeddings, margin)
    return loss

triplet_loss = triplet_loss(Alpha)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#parallel_model = multi_gpu_model(model_combined, gpus=4)
model_combined.compile(loss=triplet_loss, optimizer=sgd)
#parallel_model.compile(loss=triplet_loss, optimizer=sgd)

checkpoint_callback = ModelCheckpoint(os.path.join("models/run_1/","epoch_{epoch:06d}.h5"))
tensorboard_callback = TensorBoard()
model_combined.fit(x=[image_list, text_list], y=labels, batch_size=BatchSize, epochs=200, callbacks=[checkpoint_callback, tensorboard_callback], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
print("Done!")
