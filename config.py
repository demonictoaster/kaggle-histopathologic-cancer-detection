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
	'debug': False,  # debug mode (uses less images)
	'cloud': True,  # if script to be run via paperspace.com using GPU
	'val_size': 0.1,
	'batch_size': 32,
	'epochs': 1,
	'learning_rate': 0.0001,
	'val_metric': 'val_auc',  # 'val_acc' (accuracy) or 'val_auc' (AUC)
	'early_stopping': False,
	'patience': 10}

# folders
ROOT = os.path.abspath('')
OUTPUT = os.path.join(ROOT, 'output')
INPUT = os.path.join(ROOT, 'input')

# constants
IMG_SIZE = 96

# adjust some parameters if in debug mode
if params['debug']==True:
	params['val_size'] = 0.25
	params['epochs'] = 5
	params['patience'] = 2

# adjust folders if run in cloud
if params['cloud'] == True:
	OUTPUT = '/storage/kaggle_cancer_competition/output'
	INPUT = '/storage/kaggle_cancer_competition/input'

# print current parameters
print('\n--------------------------')
print('RUNNING WITH PARAMETERS:')
print('-DEBUG = %s' % params['debug'])
print('-CLOUD = %s' % params['cloud'])
print('-VALIDATION SIZE = %s' % params['val_size'])
print('-BATCH SIZE = %s' % params['batch_size'])
print('-EPOCHS = %s' % params['epochs'])
print('-LEARNING RATE = %s' % params['learning_rate'])
print('-VALIDATION METRIC = %s' % params['val_metric'])
print('-EARLY STOPPING = %s' % params['early_stopping'])
print('-PATIENCE = %s' % params['patience'])
print('--------------------------')

# some checks
if params['val_metric'] not in ['val_acc', 'val_auc']:
	raise ValueError('`val_metric` should be either `val_acc` or `val_auc`')
