import face_model
import argparse
import cv2
import sys
import numpy as np

import os
import csv
import glob

# general
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args) 

### CHANGE THIS LINE TO YOUR <INPUT> DIRECTORY OF IMAGES
input_dir = './images/'

### CHANGE THIS LINE TO YOUR <OUTPUT> DIRECTORY OF CSV'S
output_dir = './csv/'

fnames1 = glob.glob(input_dir+'*.jpeg')
fnames2 = glob.glob(input_dir+'.png')
fnames = fnames1+fnames2
fnames_count = len(fnames)

for i, fname in enumerate(fnames):
		
	img = cv2.imread(fname)

	inp = model.get_input(img)

	if inp.any():
		feat = model.get_feature(inp)
	else:
		print('No face found.')
		break

	if feat.any():

		csv_fname = fname.split('/')[-1].split('.')[0]+'.csv' 

		### COMMENT NEXT LINE OUT IF YOU DO NOT CARE ABOUT SEEING PROGRESS
		print('Writing to \'{}\' - Image {}/{}'.format(output_dir+csv_fname, i+1, fnames_count))
		
		with open(output_dir+'{}.csv'.format(csv_fname), 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(feat)

print('Done processing images.')

# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
# img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
# f2 = model.get_feature(img)
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)
# print(sim)
# diff = np.subtract(source_feature, target_feature)
# dist = np.sum(np.square(diff),1)
