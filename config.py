import numpy as np
import os


"""
NOTE:
- this config file stores global parameters

TODO:
- print parameters used
"""

# params
params = {
	'debug': False,  # debug mode (uses less images)
	'cloud': False,  # if script to be run on paperspace.com using GPU
	'val_size': 0.1,
	'batch_size': 32,
	'epochs': 1,
	'patience': 10,
	'debug_train_size': 32*20,
	'debug_val_size': 32*2,
	'debug_test_size': 32*2
	}

# folders
ROOT = os.path.abspath('')
OUTPUT = os.path.join(ROOT, 'output')
INPUT = os.path.join(ROOT, 'input')

# constants
IMG_SIZE = 96

# adjust some parameters if in debug mode
if params['debug']==True:
	params['epochs'] = 4
	params['val_size'] = 0.25

# adjust some parameters if run on cloud
if params['cloud'] == True:
	OUTPUT = '/storage/kaggle_cancer_competition/output'
	INPUT = '/storage/kaggle_cancer_competition/input'

# print current parameters
print('\n-----------------------')
print('RUNNING WITH PARAMETERS:')
print('-DEBUG = %s' % params['debug'])
print('-CLOUD = %s' % params['cloud'])
print('-VALIDATION SIZE = %s' % params['val_size'])
print('-BATCH SIZE = %s' % params['batch_size'])
print('-EPOCHS = %s' % params['epochs'])
print('-PATIENCE = %s' % params['patience'])
print('-----------------------')
