import numpy as np
import pandas as pd 
import pydicom as dicom
import os, json, sys
import scipy.ndimage
import matplotlib.pyplot as plt
import argparse
import h5py
import re

from skimage import measure, morphology
from skimage import exposure
from skimage.filters import rank
from PIL import Image

import get_segments
import augment_data
import get_model_data
import model_VGG16_M4M5M6 as m_vgg_m456
import model_VGG16 as m_vgg
import model_VGG16_aws as m_vgg_aws
import model_VGG16_aws_functional as m_vgg_aws_f
import getASPECTS


#####################################################################################

def str2bool(param):
	if param.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif param.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize(image, min_bound = -200.0, max_bound = 200.0):

	image = (image - min_bound) / (max_bound - min_bound)
	image[image>1] = 1.0
	image[image<0] = 0.0

	# print("{} -- {}".format(image.max(), image.min()))
	# print(image[150, 250])
	# print(image[250, 150])
	# print(image[1, 1])
	# print(image[1, 510])

	return image



def zero_center(image, pixel_mean):
    image = image - pixel_mean
    return image


def save_image_to_disk(img, clip_limit=0.07, image_name="image", out_dir = "/tmp"):

	image_name = image_name.replace("/","_")

	jpeg_outdir = out_dir
	if not "standard" in out_dir:

		jpeg_outdir = "{}/jpeg_images/".format(out_dir)

	if not(os.path.isdir(jpeg_outdir)):
		os.makedirs(jpeg_outdir)
	
	img_adapteq = exposure.equalize_adapthist(img, clip_limit=clip_limit)
	plt.imshow(img_adapteq, cmap=plt.cm.gray)
	
	outfile_name = "{}/{}.jpeg".format(jpeg_outdir, image_name.replace(".dcm",""))
	plt.savefig(outfile_name)
	print("Saved: {}".format(image_name))


def remove_irrelevent_hu(img, min_limit = -200, max_limit = 200):

	img[(img <= min_limit) | (img >= max_limit)] = img.min()

	return img


def read_dicom(path):

	patient_files = []
	for cur_path in os.walk(path):
		file_list = cur_path[-1]
		file_list_extended = ["{}/{}".format(cur_path[0], cur_file) for cur_file in file_list if cur_file.endswith(".dcm")]
		patient_files = patient_files + file_list_extended
		
		
	slices = []
	for patient_file in patient_files:
		try:
			slices.append((dicom.dcmread(patient_file), patient_file))
		except Exception as e:
			print("dcm read failed for: {}".format(patient_file))
			print(str(e))
			pass
	
	slices.sort(key = lambda x: int(x[0].InstanceNumber))

	try:
		slice_thickness = np.abs(slices[0][0].ImagePositionPatient[2] - slices[1][0].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0][0].SliceLocation - slices[1][0].SliceLocation)
		pass

	
	ip0,ip1 = slices[0][0].ImagePositionPatient[:2]
	for cur_slice in slices:

		if not cur_slice[0].SliceThickness:
			cur_slice[0].SliceThickness = slice_thickness
		
		assert cur_slice[0].ImagePositionPatient[0] == ip0 and cur_slice[0].ImagePositionPatient[1] == ip1, 'error'
		

	return slices


def get_pixels_hu(slices, slice_names):

	images_tuple = []
	for idx in range(len(slices)):
		try:
			images_tuple.append((slices[idx].pixel_array.astype(np.int16), slice_names[idx]))
		except Exception as e:
			print("Pixal array read failed for slice: {}".format(idx))
			print(str(e))
			pass
	
	images, image_names = zip(*images_tuple)

	intercept = slices[0].RescaleIntercept
	slope = slices[0].RescaleSlope

	for image in images:
		image[image <= -2000] = 0

		if slope != 1:
			image = slope * image.astype(np.float64)
			image = image.astype(np.int16)

		image += np.int16(intercept)

	return (images, image_names)


def resample2d(image, scan, new_spacing=[1,1]):
    
    spacing = np.array(scan[0].PixelSpacing, dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    image = image.astype(np.int16)
    print("Image shape: {}".format(image.shape))

    return image, new_spacing


def preprocess_data(scan_slices, slice_names, dir_name = "", out_dir = "/tmp", save_image = False):

	(hu_pixels, hu_slice_names) = get_pixels_hu(scan_slices, slice_names)

	processed_slices = []
	for idx in xrange(len(hu_pixels)):
		hu_pixel = hu_pixels[idx]
		processed_hu = remove_irrelevent_hu(hu_pixel)
		normalized_hu = normalize(processed_hu)

		image_name = hu_slice_names[idx].replace(dir_name, "")
		if save_image == True:
			save_image_to_disk(normalized_hu, image_name = image_name, out_dir = out_dir)
		processed_slices.append((normalized_hu, image_name))

	return processed_slices


def process_patient(patient, patients_dir, subdir_name = "/NCCT", out_dir = "/tmp", save_image = False):

	print("Processing patient: {}".format(patient))

	scan_slices_tuple = read_dicom(patient)

	scan_slices, slice_names = zip(*scan_slices_tuple)

	processed_patient_slices = preprocess_data(scan_slices, slice_names, dir_name = patients_dir, out_dir = out_dir, save_image = save_image)

	return processed_patient_slices


def process_standard(standard_images, standard_image_names, standard_dir, out_dir = "/tmp", save_image = False):

	processed_standard_slices = preprocess_data(standard_images, standard_image_names, dir_name = standard_dir, out_dir = out_dir, save_image = save_image)

	return processed_standard_slices


def create_h5_readable_array(df, patient_dir, region = "", out_dir = "/tmp"):


	region = region.lower()
	label_start = None

	if region == "left":
		region = "Left"
		label_start = "L"
	elif region == "right":
		region = "Right"
		label_start = "R"
	elif region != "all":
		print("region: {}".format(region))
		print("[create_h5_readable_array]: Provide valid brain region name!")
		return None

	
	
	all_cols = list(df.columns.values)

	text_cols = ["Age", "Gender"]

	meta_cols = None
	label_cols = None
	ref_df = None

	if region == "all":
		meta_cols = ["idx", "STIR_ID", "Age", "Gender", "Slices", "Thickness", "RightRef", "RightStart", "RightEnd", "LeftRef", "LeftStart", "LeftEnd"]
		label_cols_right = ["RC", "RL", "RIC", "RI", "RM1", "RM2", "RM3", "RM4", "RM5", "RM6"]
		label_cols_left = ["LC", "LL", "LIC", "LI", "LM1", "LM2", "LM3", "LM4", "LM5", "LM6"]	
		label_cols = label_cols_right + label_cols_left

		ref_df = df["RightRef"]

	else:
		meta_cols = ["idx", "STIR_ID", "Age", "Gender", "Slices", "Thickness", "{}Ref".format(region), "{}Start".format(region), "{}End".format(region)]
		label_cols = [col for col in all_cols if col.startswith(label_start) and col not in meta_cols]
		ref_df = df["{}Ref".format(region)]
	
	print("Label-columns: {}".format(label_cols))

	label_df = df[label_cols]
	text_df = df[text_cols]
	
	
	

	data_hu = []
 	data_text = []
 	label = []
 	patient_id = []
 	pref_slices = []
 	ref = []

	patient_given_set = set(df["STIR_ID"])
	all_ref_slices = set()

	all_patient_data = []

	for cur_dir in sorted(os.listdir(patient_dir)):

		if cur_dir.startswith('.'):
			continue

		patient = "{}/{}/".format(patient_dir, cur_dir)
		if len(patient) <= 1:
			continue

		patient_idx = cur_dir
		if patient_idx.startswith("M") and "-" not in patient_idx:
			patient_idx = "{}-{}".format(patient_idx[0], patient_idx[1:])

		if patient_idx not in patient_given_set:
			continue

		print(patient)

		# From now on, left and right should not matter for meta-data, both have same info
		if region == "all":
			region = "Right"

		p_df = df.loc[df["STIR_ID"] == patient_idx]
		slice_start_list = list(p_df["{}Start".format(region)].astype('int64'))
		slice_end_list = list(p_df["{}End".format(region)].astype('int64'))
		
		processed_patient_slices = process_patient(patient, patients_dir = patient_dir, out_dir = OUTPUT_DIR, save_image = False)
		all_patient_data = all_patient_data + processed_patient_slices

		for (p_slice, p_name) in processed_patient_slices:
			
			name = p_name.split("IM")[1]
			numbers = re.findall(r'\d+', name)
			slice_number = int(numbers[1])

			for idx in range(len(slice_start_list)):

				# print slice_number, slice_start_list[idx], slice_end_list[idx]
				if slice_number >= slice_start_list[idx] and slice_number <= slice_end_list[idx]:

					all_ref_slices.add(slice_number)

					data_hu.append(p_slice)
					patient_id.append(patient_idx)
					pref_slices.append(slice_number)
					
					index_name = "{}_{}_{}".format(patient_idx, str(slice_start_list[idx]), str(slice_end_list[idx]))
					
					#print("ref: {}".format(ref_df.iloc[:5]))
					ref.append(ref_df.loc[index_name])
					
					text_data = text_df.loc[index_name].replace("?",0)
					data_text.append(np.array(text_data).astype('uint8'))

					labels = label_df.loc[index_name]
					labels = labels.replace("Old", 2).replace("old", 2)
					labels = labels.replace("Bleed", 3).replace("bleed", 3)
					label.append(np.array(labels).astype('uint8'))


	data_hu = np.array(data_hu)
	patient_id = np.array(patient_id)
	pref_slices = np.array(pref_slices)
	data_text = np.array(data_text)
	label = np.array(label)	
	ref = np.array(ref)		

	print("image-shape: {}".format(data_hu.shape))
	print("pid-shape: {}".format(patient_id.shape))
	print("pref-slices-shape: {}".format(pref_slices.shape))
	print("textdata-shape: {}".format(data_text.shape))
	print("labels-shape: {}".format(label.shape))
	print("ref-shape: {}".format(ref.shape))

	print("\n*************************\n")
	print(label[:10])
	print("\n\n")
	print(patient_id[:10])
	print(ref[:10])
	print(pref_slices[:10])
	print(data_text[:18])
	print("\n*************************\n")

	return ((data_hu, data_text, label, patient_id, pref_slices, ref), (all_ref_slices, all_patient_data))


def save_h5(data, out_file):

	data_hu, data_text, label, pid, pslices, ref = data

	hf = h5py.File(out_file, 'w')
	hf.create_dataset('data_hu', data = data_hu)

	dt = h5py.special_dtype(vlen=str)
	hf.create_dataset('meta_pid', data = pid, dtype=dt)
	hf.create_dataset('meta_ref', data = ref, dtype=dt)
	
	hf.create_dataset('meta_slice', data = pslices)
	hf.create_dataset('data_text', data = data_text)
	hf.create_dataset('label', data = label)


	hf.close()

	print("Saved: {}".format(out_file))


def read_hdf(file_name):

	hf = h5py.File(file_name, 'r')
		
	data_hu = hf.get('data_hu')
	data_text = hf.get('data_text')
	label = hf.get('label')
	pid = hf.get('meta_pid')
	pslices = hf.get('meta_slice')
	ref = hf.get('meta_ref')

	# for i in xrange(len(label)):
	# 	print(label[i])


	print("image-shape: {}".format(data_hu.shape))
	print("pid-shape: {}".format(pid.shape))
	print("slicenumber-shape: {}".format(pslices.shape))
	print("textdata-shape: {}".format(data_text.shape))
	print("labels-shape: {}".format(label.shape))
	print("ref-shape: {}".format(ref.shape))

	return (data_hu, data_text, label, pid, pslices, ref)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Deep-ASPECTS")

	parser.add_argument('-ac', '--action', help="Action to perform (slice select/ train model/ test)", default="S", choices=["S", "SliceSelect", "R", "RefinedSliceSelect", "G", "Group", "Train", "Test"])
	parser.add_argument('-bl', '--bi_label', help="Bi-label analysis", default="T", choices=["T", "True", "F", "False"])
	
	parser.add_argument('-pd', '--patient_dir', help="Patients dicom file directory", default = None)
	parser.add_argument('-rd', '--ref_dir', help="Reference dicom file directory", default = None)
	parser.add_argument('-lf', '--labels_file', help="Labels file", default = None)
	parser.add_argument('-od', '--out_dir', help="Directory to save processed files", default = "/tmp")
	parser.add_argument('-rg', '--region', help="Left or right region of brain", default = "left", choices=["L", "R", "left", "right"])

	parser.add_argument('-md', '--model_dir', help="Directory to read models", default = None)
	parser.add_argument('-pt', '--param_tune', help="Tune hyper-params?", default = "True", choices=["T", "True", "F", "False"])
	

	args = parser.parse_args()

	PATIENTS_DIR = args.patient_dir
	REFERENCE_DIR = args.ref_dir
	OUTPUT_DIR = args.out_dir
	LABELS_FILE = args.labels_file
	ACTION = args.action
	BI_LABEL = str2bool(args.bi_label)
	REGION = args.region.lower()
	MODEL_DIR = args.model_dir
	PARAM_TUNE = str2bool(args.param_tune)


	# Avoid accidental pre-processing call
	LOCK = 1


	out_hdf = "{}/aspects_8_10_v2.h5".format(OUTPUT_DIR)


	if ACTION in ["S", "SliceSelect"]:

		FLAG = 0

		if PATIENTS_DIR is None:
			print("Provide patient data...")
			FLAG = 1

		if LABELS_FILE is None:
			print("Provide label data...")
			FLAG = 1

		if REFERENCE_DIR is None:
			print("Provide reference data...")
			FLAG = 1

		if LOCK == 1:
			print("Preprocessing is locked. Set LOCK to 0 to pre-process...")
			FLAG = 1

		if FLAG == 0:

			outfile_slices = "{}/selected_slices.json".format(OUTPUT_DIR)

			# process patients 
		 	patients = ["{}/{}/".format(PATIENTS_DIR, cur_dir) for cur_dir in os.listdir(PATIENTS_DIR) if not cur_dir.startswith('.')]
			patients.sort()
			#patients = [patients[-3]]

			
			label_text_df = pd.read_csv(LABELS_FILE)
			label_text_df.index = label_text_df["STIR_ID"]
			text_df = label_text_df[["Age", "Gender"]]
			label_df = label_text_df.drop(["STIR_ID", "Age", "Gender"], axis=1)


			(standard_silces, standard_slice_names) = get_segments.get_standard_slices(REFERENCE_DIR)
			standard_first, standard_last = zip(*standard_silces)
			standard_first_name, standard_last_name = zip(*standard_slice_names)
			
			standard_first_processed = process_standard(standard_first, standard_first_name, standard_dir = REFERENCE_DIR, out_dir = "{}/standard/".format(OUTPUT_DIR))
			standard_last_processed = process_standard(standard_last, standard_last_name, standard_dir = REFERENCE_DIR, out_dir = "{}/standard/".format(OUTPUT_DIR))

		 	processed_patient_slices = None
		 	relevent_patient_slices = []

		 	model_data_hu = []
		 	model_data_text = []
		 	model_label = []
		 	model_patient_id = []
		 	model_slice_number = []

		 	for patient in patients:

				if len(patient) <= 1:
					continue
				
				processed_patient_slices = process_patient(patient, patients_dir = PATIENTS_DIR, out_dir = OUTPUT_DIR, save_image = False)
				cur_patient_slices, cur_patient_slice_idx = get_segments.select_slices(processed_patient_slices, standard_first_processed, standard_last_processed)
				relevent_patient_slices.append(cur_patient_slices)

				p_start, p_end = cur_patient_slices
				p_start_idx, p_end_idx = cur_patient_slice_idx

				for (p_slice, p_name) in processed_patient_slices:
					
					name = p_name.split("IM")[1]
					numbers = re.findall(r'\d+', name)
					slice_number = int(numbers[1])

					if slice_number >= p_start_idx and slice_number <= p_end_idx:	

						patient_idx = p_name.split("/")[1].strip()
						if patient_idx.startswith("M") and "-" not in patient_idx:
							patient_idx = "{}-{}".format(patient_idx[0], patient_idx[1:])

						model_data_hu.append(p_slice)
						model_patient_id.append(patient_idx)
						model_slice_number.append(slice_number)
						
						text_data = text_df.loc[patient_idx].replace("?",0)
						model_data_text.append(np.array(text_data).astype('uint8'))

						labels = label_df.loc[patient_idx]
						labels = labels.replace("Old", -1).replace("old", -1)
						model_label.append(np.array(labels).astype('uint8'))

			model_data_hu = np.array(model_data_hu)
			model_patient_id = np.array(model_patient_id)
			model_slice_number = np.array(model_slice_number)
			model_data_text = np.array(model_data_text)
			model_label = np.array(model_label)			

			print("image-shape: {}".format(model_data_hu.shape))
			print("pid-shape: {}".format(model_patient_id.shape))
			print("slicenumber-shape: {}".format(model_slice_number.shape))
			print("textdata-shape: {}".format(model_data_text.shape))
			print("labels-shape: {}".format(model_label.shape))
			

			hf = h5py.File(out_hdf, 'w')
			hf.create_dataset('data_hu', data = model_data_hu)

			dt = h5py.special_dtype(vlen=str)
			hf.create_dataset('meta_pid', data = model_patient_id, dtype=dt)
			
			hf.create_dataset('meta_slice', data = model_slice_number)
			hf.create_dataset('data_text', data = model_data_text)
			hf.create_dataset('label', data = model_label)

			hf.close()

			with open(outfile_slices, 'w') as outfile:	
				json.dump(relevent_patient_slices, outfile, sort_keys=True, indent=4)


	elif ACTION in ["R", "RefinedSliceSelect"]:


		FLAG = 0

		if PATIENTS_DIR is None:
			print("Provide patient data...")
			FLAG = 1

		if LABELS_FILE is None:
			print("Provide label data...")
			FLAG = 1


		if FLAG == 0:

			label_text_df = pd.read_csv(LABELS_FILE)
			col_names = list(label_text_df.columns.values)
			common_cols = ["STIR_ID", "Age", "Gender", "Slices", "Thickness"]	

			if BI_LABEL == "T" or BI_LABEL == True:

				left_cols = common_cols + [col for col in col_names if col.startswith("L")]
				right_cols = common_cols + [col for col in col_names if col.startswith("R")]

				print("Left columns: {}".format(left_cols))
				print("Right columns: {}".format(right_cols))

				label_text_df_left = label_text_df[left_cols]
				label_text_df_right = label_text_df[right_cols]

				label_text_df_left["LeftStart"] = label_text_df_left["LeftStart"].replace("", 0).fillna(0).astype(int)
				label_text_df_left["LeftEnd"] = label_text_df_left["LeftEnd"].replace("", 0).fillna(0).astype(int)

				label_text_df_right["RightStart"] = label_text_df_right["RightStart"].replace("", 0).fillna(0).astype(int)
				label_text_df_right["RightEnd"] = label_text_df_right["RightEnd"].replace("", 0).fillna(0).astype(int)


				label_text_df_left["idx"] = label_text_df_left["STIR_ID"] + '_' + label_text_df_left["LeftStart"].astype(str) + '_' + label_text_df_left["LeftEnd"].astype(str)
				label_text_df_left.index = label_text_df_left["idx"]


				label_text_df_right["idx"] = label_text_df_right["STIR_ID"] + '_' + label_text_df_right["RightStart"].astype(str) + '_' + label_text_df_right["RightEnd"].astype(str)
				label_text_df_right.index = label_text_df_right["idx"]

				print("****")
				print(label_text_df_left.iloc[:3])
				print(label_text_df_right.iloc[:3])
				print("****")

				#######################################
				#left_data_hu, left_data_text, left_label, left_patient_id, left_slice_number, left_ref
				#right_data_hu, right_data_text, right_label, right_patient_id, right_slice_number, right_ref

				left_data, non_ref_left = create_h5_readable_array(label_text_df_left, PATIENTS_DIR, region = "Left", out_dir = OUTPUT_DIR)
				right_data, non_ref_right = create_h5_readable_array(label_text_df_right, PATIENTS_DIR, region = "Right", out_dir = OUTPUT_DIR)

				
				#######################################
				# Update this part ******
				# Separate train,dev and test

				perm_file_name = "{}/saved_permutation_alldata_left.txt".format(OUTPUT_DIR)
				train_data_left, dev_data_left, test_data_left = get_model_data.get_train_dev_test(left_data, perm_file_name)

				perm_file_name = "{}/saved_permutation_alldata_right.txt".format(OUTPUT_DIR)
				train_data_right, dev_data_right, test_data_right = get_model_data.get_train_dev_test(right_data, perm_file_name)

				# Save left data
				out_file = "{}/aspects_8_10_v3_2_left_train.h5".format(OUTPUT_DIR)
				save_h5(train_data_left, out_file)

				out_file = "{}/aspects_8_10_v3_2_left_dev.h5".format(OUTPUT_DIR)
				save_h5(dev_data_left, out_file)

				out_file = "{}/aspects_8_10_v3_2_left_test.h5".format(OUTPUT_DIR)
				save_h5(test_data_left, out_file)

				# Save right data
				out_file = "{}/aspects_8_10_v3_2_right_train.h5".format(OUTPUT_DIR)
				save_h5(train_data_right, out_file)

				out_file = "{}/aspects_8_10_v3_2_right_dev.h5".format(OUTPUT_DIR)
				save_h5(dev_data_right, out_file)

				out_file = "{}/aspects_8_10_v3_2_right_test.h5".format(OUTPUT_DIR)
				save_h5(test_data_right, out_file)

				
				# Save non-ref slices


				all_ref_left, all_patient_data_left = non_ref_left
				all_ref_right, all_patient_data_right = non_ref_right

				all_ref = all_ref_left | all_ref_right

				non_ref_slices = []
				non_ref_pname = []
				
				# Patient data is same for left and right - one loop is sufficient
				for (p_slice, p_name) in all_patient_data_left:
				
					name = p_name.split("IM")[1]
					numbers = re.findall(r'\d+', name)
					slice_number = int(numbers[1])

					if slice_number not in all_ref:
						non_ref_slices.append(p_slice)
						non_ref_pname.append(p_name)

				non_ref_slices = np.array(non_ref_slices)
				non_ref_pname = np.array(non_ref_pname)


				hf = h5py.File("{}/non_ref_slices.h5".format(OUTPUT_DIR), 'w')
				hf.create_dataset('data_hu', data = non_ref_slices)

				dt = h5py.special_dtype(vlen=str)
				hf.create_dataset('slice_name', data = non_ref_pname, dtype=dt)
				
				hf.close()

			else:

				print("Creating refined whole data-set ....")

				label_text_df["LeftStart"] = label_text_df["LeftStart"].replace("", 0).fillna(0).astype(int)
				label_text_df["LeftEnd"] = label_text_df["LeftEnd"].replace("", 0).fillna(0).astype(int)
				label_text_df["RightStart"] = label_text_df["RightStart"].replace("", 0).fillna(0).astype(int)
				label_text_df["RightEnd"] = label_text_df["RightEnd"].replace("", 0).fillna(0).astype(int)


				#meta_cols = ["idx", "STIR_ID", "Age", "Gender", "Slices", "Thickness", "{}Ref", "{}Start".format(region), "{}End".format(region)]
				
				label_cols_right = ["RC", "RL", "RIC", "RI", "RM1", "RM2", "RM3", "RM4", "RM5", "RM6"]
				label_cols_left = ["LC", "LL", "LIC", "LI", "LM1", "LM2", "LM3", "LM4", "LM5", "LM6"]
				
				label_df_right = label_text_df[label_cols_right]
				label_df_left = label_text_df[label_cols_left]
				

				for idx in xrange(label_text_df.shape[0]):

					RIGHT_FLAG = 0
					LEFT_FLAG = 0
					
					if 1 in label_df_right.iloc[idx]:
						RIGHT_FLAG = 1

					if 1 in label_df_left.iloc[idx]:
						LEFT_FLAG = 1

					if LEFT_FLAG == 1 and RIGHT_FLAG == 1:
						print("[WARNING!] Both left and right brain have stroke:\n{}".format(label_text_df.iloc[idx]))
					
					else:

						if RIGHT_FLAG == 1:
							label_text_df["LeftStart"].iloc[idx] = label_text_df["RightStart"].iloc[idx]
							label_text_df["LeftEnd"].iloc[idx] = label_text_df["RightEnd"].iloc[idx]
							label_text_df["LeftRef"].iloc[idx] = label_text_df["RightRef"].iloc[idx]

						elif LEFT_FLAG == 1:
							label_text_df["RightStart"].iloc[idx] = label_text_df["LeftStart"].iloc[idx]
							label_text_df["RightEnd"].iloc[idx] = label_text_df["LeftEnd"].iloc[idx]
							label_text_df["RightRef"].iloc[idx] = label_text_df["LeftRef"].iloc[idx]
							

				label_text_df["idx"] = label_text_df["STIR_ID"] + '_' + label_text_df["RightStart"].astype(str) + '_' + label_text_df["RightEnd"].astype(str)
				label_text_df.index = label_text_df["idx"]

				#######################################
				
				whole_data, non_ref_whole = create_h5_readable_array(label_text_df, PATIENTS_DIR, region = "all", out_dir = OUTPUT_DIR)
				

				perm_file_name = "{}/saved_permutation_alldata.txt".format(OUTPUT_DIR)
				train_data, dev_data, test_data = get_model_data.get_train_dev_test(whole_data, perm_file_name)

				# Save left data
				out_file = "{}/aspects_8_10_v3_2_all_train.h5".format(OUTPUT_DIR)
				save_h5(train_data, out_file)

				out_file = "{}/aspects_8_10_v3_2_all_dev.h5".format(OUTPUT_DIR)
				save_h5(dev_data, out_file)

				out_file = "{}/aspects_8_10_v3_2_all_test.h5".format(OUTPUT_DIR)
				save_h5(test_data, out_file)


			########################################
			

	elif ACTION in ["G", "Group"]:


		######
		in_file = "{}/aspects_8_10_v3_2_left_train.h5".format(OUTPUT_DIR)
		left_train = read_hdf(in_file)

		
		# Save right data
		in_file = "{}/aspects_8_10_v3_2_right_train.h5".format(OUTPUT_DIR)
		right_train = read_hdf(in_file)

		in_file = "{}/aspects_8_10_v3_2_right_dev.h5".format(OUTPUT_DIR)
		right_dev = read_hdf(in_file)

		in_file = "{}/aspects_8_10_v3_2_right_test.h5".format(OUTPUT_DIR)
		right_test = read_hdf(in_file)

		label_names = ["M4", "M5", "M6"]

		# Left
		# Get augmented data for train-set, also reduce labels to desired values
		get_model_data.save_augmented_regional(left_train, right_train, ref_start = 5, ref_end = 8, label_names = label_names, out_dir = OUTPUT_DIR)


		
		######

		# hf = h5py.File(out_hdf, 'r')
		
		# data_hu = hf.get('data_hu')
		# data_text = hf.get('data_text')
		# label = hf.get('label')
		# pid = hf.get('meta_pid')
		# pslices = hf.get('meta_slice')
		# ref = hf.get('meta_ref')


		# print("image-shape: {}".format(data_hu.shape))
		# print("pid-shape: {}".format(pid.shape))
		# print("slicenumber-shape: {}".format(pslices.shape))
		# print("textdata-shape: {}".format(data_text.shape))
		# print("labels-shape: {}".format(label.shape))

		# data = (data_hu, data_text, label, pid, pslices, ref)
		

		
		# label_names = ["M4", "M5", "M6"] 	#"C", "L", "IC", "M1", "M3", "M3", 
		# #label_names = None
		
		# get_model_data.get_left_right_data(data, ref_start = 5, ref_end = 8, label_names = label_names, out_dir = OUTPUT_DIR)

		#######################################




	elif ACTION == "Train":

		label_names = ["M4", "M5", "M6"]

		if BI_LABEL == "T" or BI_LABEL == True:

			if REGION in ["l", "left"]:

				#train_data, dev_data, test_data = get_model_data.read_data(OUTPUT_DIR, region = "left", mirror_read = True)
				#m_vgg.get_model(train_data, dev_data, test_data, OUTPUT_DIR, region = "left")

				train_data = get_model_data.read_data(OUTPUT_DIR, region = "left", mirror_read = True)

				dev_file = "{}/aspects_8_10_v3_2_left_dev.h5".format(OUTPUT_DIR)
				left_dev = read_hdf(dev_file)

				dev_data, dev_mirror = get_model_data.get_regional_data(left_dev, ref_start=5, ref_end=8, label_names = label_names, region = "left", datatype = "Dev")
				if dev_mirror is not None:
					print("Found value in mirror for dev!")

				
				m_vgg_aws.get_model(train_data, dev_data, OUTPUT_DIR, region = "left", param_tune = PARAM_TUNE)	#, test_data

			else:

				train_data = get_model_data.read_data(OUTPUT_DIR, region = "right", mirror_read = True)

				dev_file = "{}/aspects_8_10_v3_2_right_dev.h5".format(OUTPUT_DIR)
				right_dev = read_hdf(dev_file)

				dev_data, dev_mirror = get_model_data.get_regional_data(right_dev, ref_start=5, ref_end=8, label_names = label_names, region = "right", datatype = "Dev")
				if dev_mirror is not None:
					print("Found value in mirror for dev!")

				m_vgg_aws.get_model(train_data, dev_data, OUTPUT_DIR, region = "right", param_tune = PARAM_TUNE)	#, test_data

				
		else:

			train_file = "{}/aspects_8_10_v3_2_all_train.h5".format(OUTPUT_DIR)
			train_data = read_hdf(train_file)

			train_data, train_mirror = get_model_data.get_regional_data(train_data, ref_start=5, ref_end=8, label_names = label_names, datatype = "train_all")

			dev_file = "{}/aspects_8_10_v3_2_all_dev.h5".format(OUTPUT_DIR)
			dev_data = read_hdf(dev_file)

			dev_data, dev_mirror = get_model_data.get_regional_data(dev_data, ref_start=5, ref_end=8, label_names = label_names, datatype = "Dev")
			
			if dev_mirror is not None:
				print("Found value in mirror for dev!")


			m_vgg_aws_f.get_model(train_data, dev_data, OUTPUT_DIR, param_tune = PARAM_TUNE)



			
			# label_names = ["M4", "M5", "M6"] 	#"C", "L", "IC", "M1", "M3", "M3", 
			# #label_names = None
			# train_data, dev_data, test_data = get_model_data.get_regional_data_old(data, ref_start = 5, ref_end = 8, label_names = label_names)



			# m_vgg_aws.get_model(train_data, dev_data, test_data, OUTPUT_DIR)
			#############################

			## Augment data
			
			# data_hu = data_hu[:10]
			# # data_text = data_text[:20]
			# label = label[:3]

			# datagen = augment_data.get_img_generator()
			# augment_data.save_augmented_img(data_hu, datagen)

			#augment_data.flip_image(data_hu[7])
			#############################

			#sys.exit()
			
			# mean_hu = np.mean(data_hu)
			# centered_data_hu = data_hu - mean_hu

			# #save_image_to_disk(data_hu[4], image_name="img1")
			# #save_image_to_disk(centered_data[4])

			# m_vgg.get_model(data_hu, data_text, label, pid, pslices, OUTPUT_DIR)

	elif ACTION == "Test":

		label_names = ["M4", "M5", "M6"]

		if MODEL_DIR is None or not(os.path.isdir(MODEL_DIR)):
			print("Model directory None or not found...")

		else:

			if BI_LABEL == "T" or BI_LABEL == True:

				if REGION in ["l", "left"]:
					
					left_data = get_model_data.read_test_data(out_dir = OUTPUT_DIR, region = "left")

					left_data, left_data_mirror = get_model_data.get_regional_data(left_data, ref_start=5, ref_end=8, label_names = label_names, region = "left", datatype = "test")
					getASPECTS.get_aspects(left_data, region = "left", model_dir = MODEL_DIR, data_type = "test")

				elif REGION in ["r", "right"]:
					right_data = get_model_data.read_test_data(out_dir = OUTPUT_DIR, region = "right")

					right_data, right_data_mirror = get_model_data.get_regional_data(right_data, ref_start=5, ref_end=8, label_names = label_names, region = "right", datatype = "test")
					getASPECTS.get_aspects(right_data, region = "right", model_dir = MODEL_DIR, data_type = "test")
				
			else:

				print("Test functional model")

				data = get_model_data.read_test_data_functional(out_dir = OUTPUT_DIR)
				data, data_mirror = get_model_data.get_regional_data(data, ref_start=5, ref_end=8, label_names = label_names, datatype = "test")

				getASPECTS.get_aspects_functional(data, model_dir = MODEL_DIR, data_type = "test")
		


		








		
		
 	






