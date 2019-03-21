import numpy as np
import pandas as pd 
import os, json, sys
import scipy.ndimage
from skimage import exposure
import matplotlib.pyplot as plt
import re
import h5py

LABEL_MAP = {
	"RC": 	0,
	"RL": 	1,
	"RIC": 	2,
	"RI":	3,
	"RM1":	4,
	"RM2":	5,
	"RM3":	6,
	"RM4":	7,
	"RM5":	8,
	"RM6":	9,
	"LC":	0,
	"LL":	1,
	"LIC":	2,
	"LI":	3,
	"LM1":	4,
	"LM2":	5,
	"LM3":	6,
	"LM4":	7,
	"LM5":	8,
	"LM6":	9
}


LABEL_MAP_ALL = {
	"RC": 	0,
	"RL": 	1,
	"RIC": 	2,
	"RI":	3,
	"RM1":	4,
	"RM2":	5,
	"RM3":	6,
	"RM4":	7,
	"RM5":	8,
	"RM6":	9,
	"LC":	10,
	"LL":	11,
	"LIC":	12,
	"LI":	13,
	"LM1":	14,
	"LM2":	15,
	"LM3":	16,
	"LM4":	17,
	"LM5":	18,
	"LM6":	19
}


REGION_TO_REF_MAP = {
	
	"I": (1,2),
	"C": (3,4),
	"L": (3,4),
	"IC": (3,4),
	"M1": (1,4),
	"M2": (1,4),
	"M3": (1,4),
	"M4": (5,8),
	"M5": (5,8),
	"M6": (5,8)
}



def get_regional_data_old(data, ref_start, ref_end, label_names = None):

	data_hu, data_text, label, pid, pslices, ref = data

	if label_names is not None:

		label_indices = []
		for lbl in label_names:
			
			label_R = "R{}".format(lbl)
			label_indices.append(LABEL_MAP_ALL[label_R]) 

			label_L = "L{}".format(lbl)
			label_indices.append(LABEL_MAP_ALL[label_L])

		label_indices = sorted(label_indices)
		print("Selected label indices: {}".format(label_indices))

		label = label[:, label_indices]
	
	region_data_hu = []
	region_data_text = []
	region_label = []
	region_pid = []
	region_pslices = []
	region_ref = []

	for idx in xrange(len(data_hu)):

		numbers = re.findall(r'\d+', ref[idx])
		ref_number = int(numbers[0])

		if ref_number >= ref_start and ref_number <= ref_end:
			region_data_hu.append(data_hu[idx])
			region_data_text.append(data_text[idx])
			region_label.append(label[idx])
			region_pid.append(pid[idx])
			region_pslices.append(pslices[idx])
			region_ref.append(ref[idx])

	region_data_hu = np.array(region_data_hu)
	region_data_text = np.array(region_data_text)
	region_label = np.array(region_label)
	region_pid = np.array(region_pid)
	region_pslices = np.array(region_pslices)
	region_ref = np.array(region_ref)

	# Shuffle array
	data_size = region_data_hu.shape[0]
	permutation = list(np.random.permutation(data_size))


	shuffled_data_hu = np.take(region_data_hu, permutation, axis = 0)
	shuffled_data_text = np.take(region_data_text, permutation, axis = 0)
	shuffled_labels = np.take(region_label, permutation, axis = 0)
	
	shuffled_pid = np.take(region_pid, permutation, axis = 0)
	shuffled_pslices = np.take(region_pslices, permutation, axis = 0)
	shuffled_ref = np.take(region_ref, permutation, axis = 0)



	# Split train, dev and test set data
	split_1 = int(0.8 * data_size)
	split_2 = int(0.9 * data_size)

	train_data_hu = shuffled_data_hu[:split_1, :]
	dev_data_hu = shuffled_data_hu[split_1:split_2, :]
	test_data_hu = shuffled_data_hu[split_2:, :]

	train_data_text = shuffled_data_text[:split_1, :]
	dev_data_text = shuffled_data_text[split_1:split_2, :]
	test_data_text = shuffled_data_text[split_2:, :]

	train_labels = shuffled_labels[:split_1, :]
	dev_labels = shuffled_labels[split_1:split_2, :]
	test_labels = shuffled_labels[split_2:, :]

	train_pid = shuffled_pid[:split_1]
	dev_pid = shuffled_pid[split_1:split_2]
	test_pid = shuffled_pid[split_2:]

	train_pslices = shuffled_pslices[:split_1]
	dev_pslices = shuffled_pslices[split_1:split_2]
	test_pslices = shuffled_pslices[split_2:]

	train_ref = shuffled_ref[:split_1]
	dev_ref = shuffled_ref[split_1:split_2]
	test_ref = shuffled_ref[split_2:]

	train_data = (train_data_hu, train_data_text, train_labels, train_pid, train_pslices, train_ref)
	dev_data = (dev_data_hu, dev_data_text, dev_labels, dev_pid, dev_pslices, dev_ref)
	test_data = (test_data_hu, test_data_text, test_labels, test_pid, test_pslices, test_ref)

	return train_data, dev_data, test_data


def get_regional_data(data, ref_start, ref_end, label_names = None, region = None, datatype = ""):


	data_hu, data_text, label, pid, pslices, ref = data
	
	if region:
		region = region.lower()
	
	if datatype:
		datatype = datatype.lower()

	if label_names is not None:

		print("Label_names: {}".format(label_names))
		print("Region: {}".format(region))

		label_indices = []
		for lbl in label_names:

			label_R = "R{}".format(lbl)
			label_L = "L{}".format(lbl)

			if region == "right":
				label_indices.append(LABEL_MAP[label_R]) 

			elif region == "left":
				label_indices.append(LABEL_MAP[label_L])

			else:
				label_indices.append(LABEL_MAP_ALL[label_R]) 
				label_indices.append(LABEL_MAP_ALL[label_L]) 

		label_indices = sorted(label_indices)
		print("Selected label indices: {}".format(label_indices))

		label = label[:, label_indices]


	region_data_hu = []
	region_data_text = []
	region_label = []
	region_pid = []
	region_pslices = []
	region_ref = []

	for idx in xrange(data_hu.shape[0]):

		numbers = re.findall(r'\d+', ref[idx])
		ref_number = int(numbers[0])

		if ref_number >= ref_start and ref_number <= ref_end:
			region_data_hu.append(data_hu[idx])
			region_data_text.append(data_text[idx])
			region_label.append(label[idx])
			region_pid.append(pid[idx])
			region_pslices.append(pslices[idx])
			region_ref.append(ref[idx])

			# print(data_text[idx])
			# print(label[idx])
			# print(pid[idx])
			# print(pslices[idx])
			# print(ref[idx])

	
	region_data_hu = np.array(region_data_hu)
	region_data_text = np.array(region_data_text)
	region_label = np.array(region_label)
	region_pid = np.array(region_pid)
	region_pslices = np.array(region_pslices)
	region_ref = np.array(region_ref)

	# print("*****************\n")
	# for i in xrange(len(region_label)):
	# 	print(region_label[i])

	# print(datatype)
	# print(region_data_text[9])
	# print(region_label[9])
	# print(region_pid[9])
	# print(region_pslices[9])
	# print(region_ref[9])

	# print(region_data_text[39])
	# print(region_label[39])
	# print(region_pid[39])
	# print(region_pslices[39])
	# print(region_ref[39])

	
	if datatype == "train":

		region_indices = []
		zero_indices = []
		mirror_hu = []
		mirror_label = []

		label_size = region_label.shape[1]

		for idx in xrange(len(region_data_hu)):

			max_label = np.max(region_label[idx])
			
			if max_label == 0:
				zero_indices.append(idx)	

			elif (1 in region_label[idx]) and (max_label < 3):
				
				region_indices.append(idx)
				
				img_inverted = np.fliplr(region_data_hu[idx])
				mirror_hu.append(img_inverted)
				mirror_label.append(region_label[idx])


		print("Region indices len {}: {}".format(len(region_indices), region_indices))
		print(region_label[region_indices[0]])
		print(region_label[region_indices[5]])

		print("Zero indices len {}: {}".format(len(zero_indices), zero_indices))

		region_indices_nz = region_indices
		region_indices = region_indices + zero_indices
		region_indices = sorted(region_indices)


		# Get new data
		region_data_hu_new = region_data_hu[region_indices]
		region_data_text_new = region_data_text[region_indices]
		region_label_new = region_label[region_indices]
		region_pid_new = region_pid[region_indices]
		region_pslices_new = region_pslices[region_indices]
		region_ref_new = region_ref[region_indices]
		
		# Add mirror
		mirror_hu = np.array(mirror_hu)
		mirror_label = np.array(mirror_label)
		mirror_text = region_data_text[region_indices_nz]
		mirror_pid = region_pid[region_indices_nz]
		mirror_pslices = region_pslices[region_indices_nz]
		mirror_ref = region_ref[region_indices_nz]

		
		# print(region_label)
		# print("\n\n Left:")
		# print(region_label_new)

		
		# print("\n\n Region mirror right")
		# print(mirror_label)

		data = (region_data_hu_new, region_data_text_new, region_label_new, region_pid_new, region_pslices_new, region_ref_new)
		mirror_data = (mirror_hu, mirror_text, mirror_label, mirror_pid, mirror_pslices, mirror_ref)

		return (data, mirror_data)


	data = (region_data_hu, region_data_text, region_label, region_pid, region_pslices, region_ref)



	return (data, None)



def save_augmented_regional(left_data, right_data, ref_start, ref_end, label_names = None, out_dir = "/tmp"):

	
	left_data, mirror_right = get_regional_data(left_data, ref_start = ref_start, ref_end = ref_end, label_names = label_names, region = "left", datatype = "train")
	(region_data_hu_left, region_data_text_left, region_label_left, region_pid_left, region_pslices_left, region_ref_left) = left_data

	right_data, mirror_left = get_regional_data(right_data, ref_start = ref_start, ref_end = ref_end, label_names = label_names, region = "right", datatype = "train")
	(region_data_hu_right, region_data_text_right, region_label_right, region_pid_right, region_pslices_right, region_ref_right) = right_data

	if mirror_left is None or mirror_right is None:
		print("[WARNING!] Mirror is None for train data")

	else:

		mirror_hu_right, mirror_text_right, mirror_label_right, mirror_pid_right, mirror_pslices_right, mirror_ref_right = mirror_right
		mirror_hu_left, mirror_text_left, mirror_label_left, mirror_pid_left, mirror_pslices_left, mirror_ref_left = mirror_left
		
		# Save data
		out_hdf_left = "{}/aspects_8_10_v3_2_left_train_rs{}_re{}.h5".format(out_dir, ref_start, ref_end)

		hf = h5py.File(out_hdf_left, 'w')
		hf.create_dataset('data_hu', data = region_data_hu_left)

		dt = h5py.special_dtype(vlen=str)
		hf.create_dataset('meta_pid', data = region_pid_left, dtype=dt)
		hf.create_dataset('meta_ref', data = region_ref_left, dtype=dt)
		
		hf.create_dataset('meta_slice', data = region_pslices_left)
		hf.create_dataset('data_text', data = region_data_text_left)
		hf.create_dataset('label', data = region_label_left)

		hf.close()


		out_hdf_right = "{}/aspects_8_10_v3_2_right_train_rs{}_re{}.h5".format(out_dir, ref_start, ref_end)

		hf = h5py.File(out_hdf_right, 'w')
		hf.create_dataset('data_hu', data = region_data_hu_right)

		dt = h5py.special_dtype(vlen=str)
		hf.create_dataset('meta_pid', data = region_pid_right, dtype=dt)
		hf.create_dataset('meta_ref', data = region_ref_right, dtype=dt)
		
		hf.create_dataset('meta_slice', data = region_pslices_right)
		hf.create_dataset('data_text', data = region_data_text_right)
		hf.create_dataset('label', data = region_label_right)

		hf.close()


		out_hdf_mirror_left = "{}/aspects_8_10_v3_2_left_train_mirror_rs{}_re{}.h5".format(out_dir, ref_start, ref_end)
		
		hf = h5py.File(out_hdf_mirror_left, 'w')
		hf.create_dataset('data_hu', data = mirror_hu_left)

		dt = h5py.special_dtype(vlen=str)
		hf.create_dataset('meta_pid', data = mirror_pid_left, dtype=dt)
		hf.create_dataset('meta_ref', data = mirror_ref_left, dtype=dt)
		
		hf.create_dataset('meta_slice', data = mirror_pslices_left)
		hf.create_dataset('data_text', data = mirror_text_left)
		hf.create_dataset('label', data = mirror_label_left)

		hf.close()


		out_hdf_mirror_right = "{}/aspects_8_10_v3_2_right_train_mirror_rs{}_re{}.h5".format(out_dir, ref_start, ref_end)

		hf = h5py.File(out_hdf_mirror_right, 'w')
		hf.create_dataset('data_hu', data = mirror_hu_right)

		dt = h5py.special_dtype(vlen=str)
		hf.create_dataset('meta_pid', data = mirror_pid_right, dtype=dt)
		hf.create_dataset('meta_ref', data = mirror_ref_right, dtype=dt)
		
		hf.create_dataset('meta_slice', data = mirror_pslices_right)
		hf.create_dataset('data_text', data = mirror_text_right)
		hf.create_dataset('label', data = mirror_label_right)

		hf.close()




def get_left_right_data(data, ref_start = 5, ref_end = 8, label_names = None, out_dir = "/tmp"):

	data_hu, data_text, label, pid, pslices, ref = data
	#data_hu = data_hu 	#.astype(np.float32)

	if label_names is not None:

		label_indices = []
		for lbl in label_names:
			
			label_R = "R{}".format(lbl)
			label_indices.append(LABEL_MAP_ALL[label_R]) 

			label_L = "L{}".format(lbl)
			label_indices.append(LABEL_MAP_ALL[label_L])

		label_indices = sorted(label_indices)
		print("Selected label indices: {}".format(label_indices))

		label = label[:, label_indices]


	region_data_hu = []
	region_data_text = []
	region_label = []
	region_pid = []
	region_pslices = []
	region_ref = []

	for idx in xrange(data_hu.shape[0]):

		numbers = re.findall(r'\d+', ref[idx])
		ref_number = int(numbers[0])

		if ref_number >= ref_start and ref_number <= ref_end:
			region_data_hu.append(data_hu[idx])
			region_data_text.append(data_text[idx])
			region_label.append(label[idx])
			region_pid.append(pid[idx])
			region_pslices.append(pslices[idx])
			region_ref.append(ref[idx])

	region_data_hu = np.array(region_data_hu)
	region_data_text = np.array(region_data_text)
	region_label = np.array(region_label)
	region_pid = np.array(region_pid)
	region_pslices = np.array(region_pslices)
	region_ref = np.array(region_ref)

	region_pid = np.expand_dims(pid, axis=-1)
	region_pslices = np.expand_dims(pslices, axis=-1)
	region_ref = np.expand_dims(ref, axis=-1)


	left_indices = []
	right_indices = []
	zero_indices = []

	label_size = region_label.shape[1]

	assert(label_size % 2 == 0)

	new_label_size = label_size/2

	left_mirror_hu = []
	left_mirror_label = []

	right_mirror_hu = []
	right_mirror_label = []

	for idx in xrange(len(region_data_hu)):

		sum_left = np.sum(region_label[idx, new_label_size:])
		sum_right = np.sum(region_label[idx, :new_label_size])

		if sum_left%255 > 0 and sum_right%255 > 0:
			print("[*** WARNING ***] Found STROKE at both sides! index={}".format(idx))


		if (sum_left%255 == 0) and (sum_right%255 == 0):
			zero_indices.append(idx)	

		elif sum_left%255 > 0:
			left_indices.append(idx)
			
			img_inverted = np.fliplr(region_data_hu[idx])
			right_mirror_hu.append(img_inverted)

			label_inverted = region_label[idx, new_label_size:]
			right_mirror_label.append(label_inverted)

		elif sum_right%255 > 0:
			right_indices.append(idx)

			img_inverted = np.fliplr(region_data_hu[idx])
			left_mirror_hu.append(img_inverted)

			label_inverted = region_label[idx, :new_label_size]
			left_mirror_label.append(label_inverted)


	print("Left indices len {}: {}".format(len(left_indices), left_indices))
	print("Right indices len {}: {}".format(len(right_indices), right_indices))
	print("Zero indices len {}: {}".format(len(zero_indices), zero_indices))


	left_indices_nz = left_indices
	right_indices_nz = right_indices

	left_indices = left_indices + zero_indices
	left_indices = sorted(left_indices)

	right_indices = right_indices + zero_indices
	right_indices = sorted(right_indices)


	# Get data
	region_data_hu_left = region_data_hu[left_indices]
	region_data_hu_right = region_data_hu[right_indices]
	
	region_data_text_left = region_data_text[left_indices]
	region_data_text_right = region_data_text[right_indices]

	region_label_left = region_label[left_indices, new_label_size:]
	region_label_right = region_label[right_indices, :new_label_size]

	region_pid_left = region_pid[left_indices]
	region_pid_right = region_pid[right_indices]

	region_pslices_left = region_pslices[left_indices]
	region_pslices_right = region_pslices[right_indices]

	region_ref_left = region_ref[left_indices]
	region_ref_right = region_ref[right_indices]


	# Add mirror
	left_mirror_hu = np.array(left_mirror_hu)
	left_mirror_label = np.array(left_mirror_label)

	right_mirror_hu = np.array(right_mirror_hu)
	right_mirror_label = np.array(right_mirror_label)

	left_mirror_text = region_data_text[right_indices_nz]
	right_mirror_text = region_data_text[left_indices_nz]

	left_mirror_pid = region_pid[right_indices_nz]
	right_mirror_pid = region_pid[left_indices_nz]

	left_mirror_pslices = region_pslices[right_indices_nz]
	right_mirror_pslices = region_pslices[left_indices_nz]

	left_mirror_ref = region_ref[right_indices_nz]
	right_mirror_ref = region_ref[left_indices_nz]

	print(region_label)
	print("\n\n Left:")
	print(region_label_left)

	print("\n\n Right")
	print(region_label_right)

	print("\n\n Region mirror right")
	print(right_mirror_label)

	pd.DataFrame(region_label).to_csv("/tmp/labels_whole.csv")
	pd.DataFrame(region_label_left).to_csv("/tmp/labels_left.csv")
	pd.DataFrame(region_label_right).to_csv("/tmp/labels_right.csv")
	pd.DataFrame(left_mirror_label).to_csv("/tmp/labels_left_mirror.csv")
	pd.DataFrame(right_mirror_label).to_csv("/tmp/labels_right_mirror.csv")


	print("\n\n *** Left:")
	print("image-shape: {}".format(region_data_hu_left.shape))
	print("pid-shape: {}".format(region_pid_left.shape))
	print("slicenumber-shape: {}".format(region_pslices_left.shape))
	print("textdata-shape: {}".format(region_data_text_left.shape))
	print("labels-shape: {}".format(region_label_left.shape))


	print("\n\n *** Left M:")
	print("image-shape: {}".format(left_mirror_hu.shape))
	print("pid-shape: {}".format(left_mirror_pid.shape))
	print("slicenumber-shape: {}".format(left_mirror_pslices.shape))
	print("textdata-shape: {}".format(left_mirror_text.shape))
	print("labels-shape: {}".format(left_mirror_label.shape))


	print("\n\n *** Right:")
	print("image-shape: {}".format(region_data_hu_right.shape))
	print("pid-shape: {}".format(region_pid_right.shape))
	print("slicenumber-shape: {}".format(region_pslices_right.shape))
	print("textdata-shape: {}".format(region_data_text_right.shape))
	print("labels-shape: {}".format(region_label_right.shape))


	print("\n\n *** Right M:")
	print("image-shape: {}".format(right_mirror_hu.shape))
	print("pid-shape: {}".format(right_mirror_pid.shape))
	print("slicenumber-shape: {}".format(right_mirror_pslices.shape))
	print("textdata-shape: {}".format(right_mirror_text.shape))
	print("labels-shape: {}".format(right_mirror_label.shape))




	'''
	# Save data
	out_hdf_left = "{}/aspects_8_10_left.h5".format(out_dir)

	hf = h5py.File(out_hdf_left, 'w')
	hf.create_dataset('data_hu', data = region_data_hu_left)

	dt = h5py.special_dtype(vlen=str)
	hf.create_dataset('meta_pid', data = region_pid_left, dtype=dt)
	hf.create_dataset('meta_ref', data = region_ref_left, dtype=dt)
	
	hf.create_dataset('meta_slice', data = region_pslices_left)
	hf.create_dataset('data_text', data = region_data_text_left)
	hf.create_dataset('label', data = region_label_left)

	hf.close()


	out_hdf_right = "{}/aspects_8_10_right.h5".format(out_dir)

	hf = h5py.File(out_hdf_right, 'w')
	hf.create_dataset('data_hu', data = region_data_hu_right)

	dt = h5py.special_dtype(vlen=str)
	hf.create_dataset('meta_pid', data = region_pid_right, dtype=dt)
	hf.create_dataset('meta_ref', data = region_ref_right, dtype=dt)
	
	hf.create_dataset('meta_slice', data = region_pslices_right)
	hf.create_dataset('data_text', data = region_data_text_right)
	hf.create_dataset('label', data = region_label_right)

	hf.close()


	out_hdf_mirror_left = "{}/aspects_8_10_left_mirror.h5".format(out_dir)
	
	hf = h5py.File(out_hdf_mirror_left, 'w')
	hf.create_dataset('data_hu', data = left_mirror_hu)

	dt = h5py.special_dtype(vlen=str)
	hf.create_dataset('meta_pid', data = left_mirror_pid, dtype=dt)
	hf.create_dataset('meta_ref', data = left_mirror_ref, dtype=dt)
	
	hf.create_dataset('meta_slice', data = left_mirror_pslices)
	hf.create_dataset('data_text', data = left_mirror_text)
	hf.create_dataset('label', data = left_mirror_label)

	hf.close()


	out_hdf_mirror_right = "{}/aspects_8_10_right_mirror.h5".format(out_dir)

	hf = h5py.File(out_hdf_mirror_right, 'w')
	hf.create_dataset('data_hu', data = right_mirror_hu)

	dt = h5py.special_dtype(vlen=str)
	hf.create_dataset('meta_pid', data = right_mirror_pid, dtype=dt)
	hf.create_dataset('meta_ref', data = right_mirror_ref, dtype=dt)
	
	hf.create_dataset('meta_slice', data = right_mirror_pslices)
	hf.create_dataset('data_text', data = right_mirror_text)
	hf.create_dataset('label', data = right_mirror_label)

	hf.close()
	'''


def get_train_dev_test(data, permutation_file, action = None):

	data_hu, data_text, label, pid, pslices, ref = data

	# Shuffle array
	data_size = data_hu.shape[0]

	permutation = None
	#permutation_file = "{}/saved_permutation_m456.txt".format(out_dir)
	
	if os.path.isfile(permutation_file) and action == "Test":
		with open(permutation_file, "r") as p_file:
			permutation = json.load(p_file)
	else:
		permutation = list(np.random.permutation(data_size))
		with open(permutation_file, "w") as outfile:	
			json.dump(permutation, outfile)

	assert(permutation)
	assert(len(permutation) == len(data_hu))

	shuffled_data_hu = np.take(data_hu, permutation, axis = 0)
	shuffled_data_text = np.take(data_text, permutation, axis = 0)
	shuffled_labels = np.take(label, permutation, axis = 0)
	
	shuffled_pid = np.take(pid, permutation, axis = 0)
	shuffled_pslices = np.take(pslices, permutation, axis = 0)
	shuffled_ref = np.take(ref, permutation, axis = 0)



	# Split train, dev and test set data
	split_1 = int(0.8 * data_size)
	split_2 = int(0.9 * data_size)


	train_data_hu = shuffled_data_hu[:split_1]
	dev_data_hu = shuffled_data_hu[split_1:split_2]
	test_data_hu = shuffled_data_hu[split_2:]

	train_data_text = shuffled_data_text[:split_1]
	dev_data_text = shuffled_data_text[split_1:split_2]
	test_data_text = shuffled_data_text[split_2:]

	train_labels = shuffled_labels[:split_1]
	dev_labels = shuffled_labels[split_1:split_2]
	test_labels = shuffled_labels[split_2:]

	train_pid = shuffled_pid[:split_1]
	dev_pid = shuffled_pid[split_1:split_2]
	test_pid = shuffled_pid[split_2:]

	train_pslices = shuffled_pslices[:split_1]
	dev_pslices = shuffled_pslices[split_1:split_2]
	test_pslices = shuffled_pslices[split_2:]

	train_ref = shuffled_ref[:split_1]
	dev_ref = shuffled_ref[split_1:split_2]
	test_ref = shuffled_ref[split_2:]

	train_data = (train_data_hu, train_data_text, train_labels, train_pid, train_pslices, train_ref)
	dev_data = (dev_data_hu, dev_data_text, dev_labels, dev_pid, dev_pslices, dev_ref)
	test_data = (test_data_hu, test_data_text, test_labels, test_pid, test_pslices, test_ref)

	return train_data, dev_data, test_data



def read_hdf(file_name):

	hf = h5py.File(file_name, 'r')
	
	data_hu = hf.get('data_hu')
	data_text = hf.get('data_text')
	label = hf.get('label')
	pid = hf.get('meta_pid')
	pslices = hf.get('meta_slice')
	ref = hf.get('meta_ref')

	# print("image-shape: {}".format(data_hu.shape))
	# print("pid-shape: {}".format(pid.shape))
	# print("slicenumber-shape: {}".format(pslices.shape))
	# print("textdata-shape: {}".format(data_text.shape))
	# print("labels-shape: {}".format(label.shape))

	return (data_hu, data_text, label, pid, pslices, ref)


def read_data(out_dir, region = "left", mirror_read = True):


	if region.lower() == "left":

		print("\nIn Left ...\n")
		#hdf_left = "{}/aspects_8_10_left.h5".format(out_dir)
		hdf_left = "{}/aspects_8_10_v3_2_left_train_rs5_re8.h5".format(out_dir)

		data_hu, data_text, label, pid, pslices, ref = read_hdf(hdf_left)
		pid = np.expand_dims(pid, axis=-1)
		pslices = np.expand_dims(pslices, axis=-1)
		ref = np.expand_dims(ref, axis=-1)

		print("image-shape: {}".format(data_hu.shape))
		print("pid-shape: {}".format(pid.shape))
		print("slicenumber-shape: {}".format(pslices.shape))
		print("textdata-shape: {}".format(data_text.shape))
		print("labels-shape: {}".format(label.shape))

		if mirror_read == True:
			
			hdf_mirror_left = "{}/aspects_8_10_v3_2_left_train_mirror_rs5_re8.h5".format(out_dir)
			m_data_hu, m_data_text, m_label, m_pid, m_pslices, m_ref = read_hdf(hdf_mirror_left)
			m_pid = np.expand_dims(m_pid, axis=-1)
			m_pslices = np.expand_dims(m_pslices, axis=-1)
			m_ref = np.expand_dims(m_ref, axis=-1)

			print("\n\n *** Mirror data shape:")
			print("image-shape: {}".format(m_data_hu.shape))
			print("pid-shape: {}".format(m_pid.shape))
			print("slicenumber-shape: {}".format(m_pslices.shape))
			print("textdata-shape: {}".format(m_data_text.shape))
			print("labels-shape: {}\n\n".format(m_label.shape))

			data_hu = np.vstack((data_hu, m_data_hu))
			data_text = np.vstack((data_text, m_data_text))
			label = np.vstack((label, m_label))
			pid = np.vstack((pid, m_pid))
			pslices = np.vstack((pslices, m_pslices))
			ref = np.vstack((ref, m_ref))

			print("\n\n *** Final data shape:")
			print("image-shape: {}".format(data_hu.shape))
			print("pid-shape: {}".format(pid.shape))
			print("slicenumber-shape: {}".format(pslices.shape))
			print("textdata-shape: {}".format(data_text.shape))
			print("labels-shape: {}".format(label.shape))

		data = (data_hu, data_text, label, pid, pslices, ref)
		#train_data, dev_data, test_data = get_train_dev_test(data, out_dir)

		#return (train_data, dev_data, test_data)

		return data
	

	else:
		print("\n\nIn right ...\n")
		hdf_right = "{}/aspects_8_10_v3_2_right_train_rs5_re8.h5".format(out_dir)

		data_hu, data_text, label, pid, pslices, ref = read_hdf(hdf_right)
		pid = np.expand_dims(pid, axis=-1)
		pslices = np.expand_dims(pslices, axis=-1)
		ref = np.expand_dims(ref, axis=-1)

		print("image-shape: {}".format(data_hu.shape))
		print("pid-shape: {}".format(pid.shape))
		print("slicenumber-shape: {}".format(pslices.shape))
		print("textdata-shape: {}".format(data_text.shape))
		print("labels-shape: {}".format(label.shape))


		if mirror_read == True:
			hdf_mirror_right = "{}/aspects_8_10_v3_2_right_train_mirror_rs5_re8.h5".format(out_dir)

			m_data_hu, m_data_text, m_label, m_pid, m_pslices, m_ref = read_hdf(hdf_mirror_right)
			m_pid = np.expand_dims(m_pid, axis=-1)
			m_pslices = np.expand_dims(m_pslices, axis=-1)
			m_ref = np.expand_dims(m_ref, axis=-1)

			print("\n\n *** Mirror data shape:")
			print("image-shape: {}".format(m_data_hu.shape))
			print("pid-shape: {}".format(m_pid.shape))
			print("slicenumber-shape: {}".format(m_pslices.shape))
			print("textdata-shape: {}".format(m_data_text.shape))
			print("labels-shape: {}".format(m_label.shape))

			data_hu = np.vstack((data_hu, m_data_hu))
			data_text = np.vstack((data_text, m_data_text))
			label = np.vstack((label, m_label))
			pid = np.vstack((pid, m_pid))
			pslices = np.vstack((pslices, m_pslices))
			ref = np.vstack((ref, m_ref))

			print("\n\n *** Final data shape:")
			print("image-shape: {}".format(data_hu.shape))
			print("pid-shape: {}".format(pid.shape))
			print("slicenumber-shape: {}".format(pslices.shape))
			print("textdata-shape: {}".format(data_text.shape))
			print("labels-shape: {}".format(label.shape))


		data = (data_hu, data_text, label, pid, pslices, ref)
		
		return data


def read_test_data(out_dir, region):

	###################################

	data = None
	hdf_file = None

	if region in ["l", "left"]:

		print("\nReading Left ...\n")		
		hdf_file = "{}/aspects_8_10_v3_2_left_test.h5".format(out_dir)
	
	elif region in ["r", "right"]:

		print("\n\nReading right ...\n")
		hdf_file = "{}/aspects_8_10_v3_2_right_test.h5".format(out_dir)

		
	data_hu, data_text, label, pid, pslices, ref = read_hdf(hdf_file)
	# pid = np.expand_dims(pid, axis=-1)
	# pslices = np.expand_dims(pslices, axis=-1)
	# ref = np.expand_dims(ref, axis=-1)

	print("image-shape: {}".format(data_hu.shape))
	print("textdata-shape: {}".format(data_text.shape))
	print("labels-shape: {}".format(label.shape))
	print("pid-shape: {}".format(pid.shape))
	print("slicenumber-shape: {}".format(pslices.shape))
	print("ref-shape: {}".format(ref.shape))
	

	return (data_hu, data_text, label, pid, pslices, ref)



def read_test_data_functional(out_dir):

	###################################

	print("\nReading data ...\n")
	
	hdf_data = "{}/aspects_8_10_v3_2_all_test.h5".format(out_dir)

	data_hu, data_text, label, pid, pslices, ref = read_hdf(hdf_data)
	# pid = np.expand_dims(pid, axis=-1)
	# pslices = np.expand_dims(pslices, axis=-1)
	# ref = np.expand_dims(ref, axis=-1)

	print("image-shape: {}".format(data_hu.shape))
	print("pid-shape: {}".format(pid.shape))
	print("slicenumber-shape: {}".format(pslices.shape))
	print("textdata-shape: {}".format(data_text.shape))
	print("labels-shape: {}".format(label.shape))

	
	return (data_hu, data_text, label, pid, pslices, ref)
	

















	
	