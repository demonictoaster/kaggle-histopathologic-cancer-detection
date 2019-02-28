import numpy as np
import os


"""
NOTE:
- this config file stores global parameters
"""

# params
params = {
	'debug': True,  # debug mode (uses less images)
	'cloud': False,  # if script to be run on paperspace.com using GPU
	'val_size': 0.2,
	'batch_size': 32,
	'epochs': 2,
	'patience': 10,
	'debug_train_size': 32*20,
	'debug_val_size': 32*2,
	'debug_test_size': 32*2
	}

# folders
ROOT = os.path.abspath('')
OUTPUT = os.path.join(ROOT, 'output')
TRAIN = os.path.join(ROOT, 'temp', 'train')
VAL = os.path.join(ROOT, 'temp', 'val')
TEST = os.path.join(ROOT, 'input', 'test')

# constants
IMG_SIZE = 96

# make sure train set and val set are a multiple of batch size
# Keras ImageDataGenerator doesn't behave properly otherwise
N = 220025
n_batches = np.floor(N / params['batch_size'])
n_batches_train = np.round(n_batches * (1-params['val_size']))
n_batches_val = n_batches - n_batches_train
N_train = int(n_batches_train * params['batch_size'])
N_val = int(n_batches_val * params['batch_size'])

# adjust some parameters if in debug mode
if params['debug']==True:
	TRAIN = os.path.join(ROOT, 'debug', 'train')
	VAL = os.path.join(ROOT, 'debug', 'val')
	TEST = os.path.join(ROOT, 'debug', 'test')
	params['epochs'] = 10
	params['patience'] = 2
	N_train = params['debug_train_size']
	N_val = params['debug_val_size']

# adjust some parameters if run on cloud
if params['cloud'] == True:
	assert params['debug'] == False, 'Cloud version should not run in debug mode.'

	OUTPUT = '/storage/kaggle_cancer_competition/output'
	TRAIN = '/storage/kaggle_cancer_competition/temp/train'
	VAL = '/storage/kaggle_cancer_competition/temp/val'
	TEST = '/storage/kaggle_cancer_competition/temp/test'


