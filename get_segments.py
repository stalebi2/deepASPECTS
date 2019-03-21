from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import json, os, re
from pandas.io.json import json_normalize

 


def get_scores(img1, img2, method = "ssim"):

	if method == "ssim":
		return ssim(img1, img2)
	elif method == ["mse"]:
		return mse(img1, img2)
	elif method == "nrmse":
		return nrmse(img1, img2)
	
	return None


def select_slices(processed_patient_slices, standard_first_processed, standard_last_processed, threshold = 0.74):

	scores_first = {}
	scores_last = {}

	scores_first_avg = []
	scores_last_avg = []


	'''
	# Find min standard slice number
	standard_first_names = zip(*standard_first_processed)[1]
	min_first_slice = 100
	
	for name in standard_first_names:
		numbers = re.findall(r'\d+', name)
		slice_number = int(numbers[len(numbers) - 1])

		if slice_number < min_first_slice:
			min_first_slice = slice_number


	# Find max standard slice number
	standard_last_names = zip(*standard_last_processed)[1]
	max_last_slice = 0
	
	for name in standard_last_names:
		numbers = re.findall(r'\d+', name)
		slice_number = int(numbers[len(numbers) - 1])

		if slice_number > max_last_slice:
			max_last_slice = slice_number

	'''


	for (p_slice, p_name) in processed_patient_slices:

		scores_first[p_name] = []
		sum_score_first = 0.0

		for (standard_slice, standard_slice_name) in standard_first_processed:
			p_score = get_scores(standard_slice, p_slice)
			scores_first[p_name].append((p_score, standard_slice_name))
			
			sum_score_first = sum_score_first + p_score

		avg_score_first = sum_score_first/ len(standard_first_processed)
		scores_first_avg.append((avg_score_first, p_name))


		scores_last[p_name] = []
		sum_score_last = 0.0

		
		for (standard_slice, standard_slice_name) in standard_last_processed:
			p_score = get_scores(standard_slice, p_slice)
			scores_last[p_name].append((p_score, standard_slice_name))

			sum_score_last = sum_score_last + p_score

		avg_score_last = sum_score_last/ len(standard_last_processed)
		scores_last_avg.append((avg_score_last, p_name))


	scores_first_avg = sorted(scores_first_avg, key = lambda x: x[0], reverse = True)
	scores_last_avg = sorted(scores_last_avg, key = lambda x: x[0], reverse = True)

	
	idx = 2
	while scores_first_avg[idx][0] > threshold:
		idx = idx + 1

	selected_first_pairs = scores_first_avg[:idx]
	selected_first_names = zip(*selected_first_pairs)[1]

	idx = 2
	while scores_last_avg[idx][0] > threshold:
		idx = idx + 1
	
	selected_last_pairs = scores_last_avg[:idx]
	selected_last_names = zip(*selected_last_pairs)[1]


	# Find min patient slice number
	first_p_idx = 100
	first_p_name = None
	
	for full_name in selected_first_names:

		name = full_name.split("IM")[1]
		numbers = re.findall(r'\d+', name)
		slice_number = int(numbers[1])

		if slice_number < first_p_idx:		# and slice_number >= min_first_slice
			first_p_idx = slice_number
			first_p_name = full_name


	# Find max patient slice number
	last_p_idx = 0
	last_p_name = None
	
	for full_name in selected_last_names:

		name = full_name.split("IM")[1]
		numbers = re.findall(r'\d+', name)
		slice_number = int(numbers[1])

		if slice_number > last_p_idx:		# and slice_number >= min_first_slice
			last_p_idx = slice_number
			last_p_name = full_name


	print("Selected first patient: {}".format(first_p_name))
	print("Selected last patient: {}".format(last_p_name))


	# with open("/tmp/dicom_scores.json", 'w') as outfile:	
	# 	json.dump(scores_first, outfile, sort_keys=True, indent=4)		
	# 	json.dump(scores_last, outfile, sort_keys=True, indent=4)


	return ((first_p_name, last_p_name), (first_p_idx, last_p_idx))


def get_standard_slices(ref_dir, subdir_name = "/NCCT"):

	ref_images = ["{}/{}/".format(ref_dir, cur_dir) for cur_dir in os.listdir(ref_dir) if not cur_dir.startswith('.')]
	ref_images.sort()

	standard_silces = []
	standard_slice_names = []
	
	for ref_image in ref_images:

		ref_files = []
		for cur_path in os.walk(ref_image):
			file_list = cur_path[-1]
			file_list_extended = ["{}/{}".format(cur_path[0], cur_file) for cur_file in file_list if cur_file.endswith(".dcm")]
			ref_files = ref_files + file_list_extended


		slices = []
		for ref_file in ref_files:
			try:
				slices.append((dicom.dcmread(ref_file), ref_file))
			except Exception as e:
				print("dcm read failed for: {}".format(ref_file))
				print(str(e))
				pass

		slices.sort(key = lambda x: int(x[0].InstanceNumber))
		standard_silces.append((slices[0][0], slices[len(slices)-1][0]))
		standard_slice_names.append((slices[0][1], slices[len(slices)-1][1]))

		print("First standard slice: {}".format(slices[0][1]))
		print("Last standard slice: {}".format(slices[len(slices)-1][1]))


	return (standard_silces, standard_slice_names)
