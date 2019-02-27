import numpy as np
import os
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

from config import *


"""
NOTE:
- In Order to use Keras flow_from_directory() , images need to be saved in
  one subdirectory per class.

TODO:
- clear folders before copying files
- show summary with number of files in each folder
"""

INPUT = os.path.join(ROOT, 'input')
TEMP = os.path.join(ROOT, 'temp')
DEBUG = os.path.join(ROOT, 'debug')

# create folders
for f1 in [TEMP, DEBUG]:
	for f2 in ['train', 'val']:
		for f3 in ['0', '1']:
			directory = os.path.join(f1, f2, f3)
			if not os.path.exists(directory):
    			os.makedirs(directory)

# get ground truths
labels = pd.read_csv(INPUT + '/train_labels.csv')

# split data between train and val
labels_train = labels.iloc[0:N_train]
labels_val = labels.iloc[N_train:(N_train+N_val)]
f2y_train = dict(zip(labels_train['id'], labels_train['label']))
f2y_val = dict(zip(labels_val['id'], labels_val['label']))

# copy files
for row in tqdm(range(0, N_train+N_val)):
	id = labels.iloc[row]['id']
	label = labels.iloc[row]['label']
	src = os.path.join(INPUT, 'train', id) + '.tif'
	if row < N_train:
		dst = os.path.join(TEMP, 'train', str(label), id) + '.tif'
	else: 
		dst = os.path.join(TEMP, 'val', str(label), id) + '.tif'
	copyfile(src, dst)


# for debugging purpose, copy only a subset of images
N_train = params['debug_train_size']
N_val = params['debug_val_size']

for row in tqdm(range(0, N_train+N_val)):
	id = labels.iloc[row]['id']
	label = labels.iloc[row]['label']
	src = os.path.join(INPUT, 'train', id) + '.tif'
	if row < N_train:
		dst = os.path.join(DEBUG, 'train', str(label), id) + '.tif'
	else: 
		dst = os.path.join(DEBUG, 'val', str(label), id) + '.tif'
	copyfile(src, dst)

# debug test images
directory = os.path.join(DEBUG, 'test')
if not os.path.exists(directory):
	os.makedirs(directory)
N_test = params['debug_test_size']
files = os.listdir(os.path.join(INPUT, 'test'))[0:N_test]
for f in tqdm(files):
	src = os.path.join(INPUT, 'test', f)
	dst = os.path.join(DEBUG, 'test', f)
	copyfile(src, dst)
