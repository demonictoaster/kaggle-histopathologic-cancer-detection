import numpy as np
import os


"""
NOTE:
- this config file stores global parameters

TODO:
- 
"""

# params
params = {
	'debug': False,  				# debug mode (uses less images)
	'cloud': True,  				# if run on cloud GPU
	'val_size': 0.1,
	'seed_validation_split': 12,
	'batch_size': 64,
	'epochs': 12,
	'learning_rate': 0.0001,
	'weights_init': 'imagenet',  	# None or 'imagenet'
	'val_metric': 'val_acc',  		# 'val_acc' (accuracy) or 'val_auc' (AUC)
	'early_stopping': False,
	'patience': 3,
	'data_aug': True,  				# data augmentation for training
	'test_time_aug': True}  		# test time augmentation

# folders
ROOT = os.path.abspath('')
OUTPUT = os.path.join(ROOT, 'output')
INPUT = os.path.join(ROOT, 'input')

# constants
IMG_SIZE = 96

# adjust some parameters if in debug mode
if params['debug']==True:
	params['val_size'] = 0.25
	params['epochs'] = 3

# adjust folders if run in cloud
if params['cloud'] == True:
	INPUT = '/storage/kaggle_cancer_competition/input'
	OUTPUT = '/storage/kaggle_cancer_competition/output'
	
# print current parameters
print('\n---------------------------')
print('RUNNING WITH PARAMETERS:')
print('-DEBUG = %s' % params['debug'])
print('-CLOUD = %s' % params['cloud'])
print('-VALIDATION SIZE = %s' % params['val_size'])
print('-BATCH SIZE = %s' % params['batch_size'])
print('-EPOCHS = %s' % params['epochs'])
print('-LEARNING RATE = %s' % params['learning_rate'])
print('-INITIAL WEIGHTS = %s' % params['weights_init'])
print('-VALIDATION METRIC = %s' % params['val_metric'])
print('-EARLY STOPPING = %s' % params['early_stopping'])
print('-PATIENCE = %s' % params['patience'])
print('-AUGMENTATION = %s' % params['data_aug'])
print('-TEST AUGMENTATION = %s' % params['test_time_aug'])
print('---------------------------')

# some checks
if params['val_metric'] not in ['val_acc', 'val_auc']:
	raise ValueError('`val_metric` should be either `val_acc` or `val_auc`')
