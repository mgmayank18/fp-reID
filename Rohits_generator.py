import os
import random
from exceptions import NotImplementedError 

from lxml import etree
import numpy as np

# from keras.preprocessing import image
#from keras.applications.inception_v3 import preprocess_input
# from keras.applications.densenet import preprocess_input

from utils import ingest_image

# def ingest_image(img_path):
# 	img = image.load_img(img_path, target_size=(224, 224))
# 	x = image.img_to_array(img)
# 	x = np.expand_dims(x, axis=0)
# 	x = preprocess_input(x)
# 	return x


	# data_dir = '/data/VeRi'
	# train_labels = 'train_label.xml'


# from pprint import pprint
# pprint(training_images)


import numpy as np

class VeRiGenerator(object):
	'Generates data for Keras'
	def __init__(self, data_dir, train_labels_file, train_images_folder, batch_size = 32, shuffle = True):
		'Initialization'
		if shuffle == False:
			raise NotImplementedError("Shuffle = False is not yet supported")
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.data_dir = data_dir
		self.train_images_folder = train_images_folder
		
		parsed_train_labels = etree.parse(os.path.join(data_dir, train_labels_file))
		xmlroot = parsed_train_labels.getroot()
		training_images = {}

		for train_image in xmlroot[0]:
			imageName = train_image.get('imageName')
			vehicleID = train_image.get('vehicleID')
			cameraID = train_image.get('cameraID')
			colorID, typeID = train_image.get('colorID'), train_image.get('typeID')
			if vehicleID in training_images.keys():
				training_images[vehicleID]['images'].append({'imageName': imageName, 'cameraID': cameraID})
			else:
				training_images[vehicleID] = {'colorID': colorID, 'typeID': typeID, 'images':[{'imageName': imageName, 'cameraID': cameraID}]}
		
		if shuffle:
			for vehicleID in training_images.keys():
				random.shuffle(training_images[vehicleID]['images'])
		
		self.training_images = training_images
		self.per_vehicle_counters = {vehicleID: 0 for vehicleID in training_images.keys()}
		self.global_counter = 0

	
	def generate(self):
		'Generates batches of samples' 
		# Infinite loop
		while True:
			# Generate ImageIDs
			image_list = self.__get_image_names()
			X, X_metadata = self.__load_image_data(image_list)
			yield X, X_metadata

	
	def __get_image_names(self):
		'Generates ImageIds to pick'
		# Find exploration order
		aux = self.training_images.keys()
		if self.shuffle == True:
			random.shuffle(aux)
			selected_vehicles = aux[:self.batch_size//8]

		imageIDs = []

		for vehicleID in selected_vehicles:
			vehicle = self.training_images[vehicleID]
			counterval = self.per_vehicle_counters[vehicleID]
			imageIDs += [{'imageName': vehicle['images'][counterval + idx]['imageName'], 'vehicleID': vehicleID } for idx in range(8)]
			self.per_vehicle_counters[vehicleID] += 8
			if self.per_vehicle_counters[vehicleID] >= len(self.training_images[vehicleID]['images']) - 8:
				self.per_vehicle_counters[vehicleID] = 0
				random.shuffle(self.training_images[vehicleID]['images'])

		return imageIDs


	def __load_image_data(self, image_list):
		'Loads data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
		image_data = []
		for image in image_list:
			image_path = os.path.join(self.data_dir, self.train_images_folder, image['imageName'])
			image_array = np.array(ingest_image(image_path, crop="random", dimension=299))
			image_data.append(image_array)
		image_labels = [int(image_dict['vehicleID']) for image_dict in image_list]
		return np.stack(image_data), np.array(image_labels)

