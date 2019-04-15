import matplotlib
# when running on server: force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from config import *
from augmentation import *


"""
NOTES:
- freezing InceptionV3 layers leads to weird results at test time due to
  bug linked to batch normalization layers 
  (http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/)
- for training, the number of steps per epoch is set to number of examples 
  divided batch size, ignoring the remainder of the division. 
  The few images that are not used in the epoch will be used in subsequent 
  epoch since data is shuffled for training.
- for validation, we create an extra batch using the last examples needed
  to go through all examples present in the validation set (in case the number
  of validation examples is not a multiple of the batch size)
- applying image augmentation doesn't increase batch size, only replaces
  a random subset of the images within the batch
"""

K.clear_session()

train_path = os.path.join(INPUT, 'train')
test_path = os.path.join(INPUT, 'test')
	
file_label_map = pd.read_csv(INPUT + '/train_labels.csv')
if params['debug'] == True:
	file_label_map = file_label_map.iloc[0:500]

f2l_train, f2l_val = train_test_split(
	file_label_map, 
	test_size=params['val_size'],
	random_state=params['seed_validation_split'])

batch_size = params['batch_size']
epochs = params['epochs']
N_train = f2l_train.shape[0]
N_val = f2l_val.shape[0]
steps_train = N_train // batch_size  # ignore remainder images (see NOTES)
steps_val = int(np.ceil(N_val / batch_size))

#################
### daga prep ###
#################

print('\nPreparing data . . .')

def load_image(file_path):
	img = cv2.imread(os.path.join(file_path))
	img = img[:,:,::-1]  # BGR -> RGB
	return img

def batch_generator(input_path, 
					f2l, 
					batch_size, 
					mode='train',  # 'train' or 'eval'
					aug=True):

	# shuffle data at each new epoch
	# don't set seed otherwise always same batches
	if mode == 'train':
		f2l = f2l.reindex(np.random.permutation(f2l.index), copy=True)

	n_files = f2l.shape[0]
	idx = 0

	if aug==True:
		seq = get_augmenter() 

	while True:
		imgs = []
		lbls = []

		# fill the batch
		while len(imgs) < batch_size:
			f = f2l.iloc[idx, 0] + '.tif'
			f_path = os.path.join(input_path, f)
			img = load_image(f_path)
			imgs.append(img)

			lbl = f2l.iloc[idx, 1]
			lbls.append(lbl)

			idx += 1

			if idx == n_files:
				idx = 0  # in case we go through all files
				if mode == 'eval':
					break

		if aug == True:
			imgs = seq.augment_images(imgs)

		imgs = [preprocess_input(i) for i in imgs]
		imgs = np.array(imgs)
		lbls = np.array(lbls).reshape(len(lbls), 1)

		yield(imgs, lbls)

# initialize the generators
train_generator = batch_generator(
	train_path,
	f2l_train,
	batch_size,
	mode='train',
	aug=params['data_aug'])

val_generator = batch_generator(
	train_path,
	f2l_val,
	batch_size,
	mode='eval',
	aug=False)

###################
### build model ###
###################

print('\nBuilding model . . .')

# custom callback to get validation AUC at end of epoch
class auc_callback(Callback):
	def __init__(self, val_gen, val_steps, val_labels):
		self.val_gen = val_gen
		self.val_steps = val_steps
		self.labels = val_labels

	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		preds = self.model.predict_generator(
			self.val_gen, 
			self.val_steps, 
			workers=0)  # use main thread since batch_generator not thread-safe
		labels = self.labels

		auc = roc_auc_score(labels, preds)
		logs['val_auc'] = auc
		print(' - val_auc: %s' % (str(round(auc,4))))

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

log_name = 'train_log.csv'
log_file = os.path.join(OUTPUT, log_name)
os.remove(log_file) if os.path.exists(log_file) else None

callback_list = [
	auc_callback(val_generator, steps_val, f2l_val['label']),
	ModelCheckpoint(
		filepath=os.path.join(OUTPUT, 'model.h5'), 
		monitor=params['val_metric'], 
		save_best_only=True,
		mode='max'),
	CSVLogger(log_file)]

if params['early_stopping'] == True:
	callback_list.insert(
		3,
		EarlyStopping(
		monitor=params['val_metric'],
		patience=params['patience']))

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
	X_input = Input(shape=input_shape)
	base_model = DenseNet169(
		input_tensor=X_input,
		include_top=False, 
		weights=params['weights_init'])

	# if params['debug']==True:
	# 	for l in base_model.layers:
	#     	l.trainable = False

	X = base_model(X_input)
	X1 = GlobalMaxPooling2D()(X)
	X2 = GlobalAveragePooling2D()(X)
	X3 = Flatten()(X)
	X = Concatenate(axis=-1)([X1, X2, X3])
	X = Dropout(0.5)(X)
	X_output = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X_output)

	model.compile(
		optimizer=Adam(params['learning_rate']),
		loss='binary_crossentropy',
		metrics=['acc'])

	return model

model = build_model()
model.summary()
architecture = model.layers[1].name

###################
### train model ###
###################

print('\nTraining model . . .')

ts = time.time()

model.fit_generator(
	generator=train_generator,
	steps_per_epoch= steps_train,
	epochs=epochs,
	callbacks=callback_list,
	validation_data=val_generator,
	validation_steps=steps_val,
	use_multiprocessing=True,
	verbose=2)

time_train = np.round((time.time() - ts) / 60, 2)

if params['debug'] == False:
	# make architecture folder and export subfolder
	arc_folder  = os.path.join(OUTPUT, architecture)  # e.g. './NasNet'
	os.makedirs(arc_folder) if not os.path.isdir(arc_folder) else None

	today = datetime.datetime.now()
	exp_folder = today.strftime('%y%m%d') + '_' + today.strftime("%H%M")
	exp_folder = os.path.join(arc_folder, exp_folder)
	os.mkdir(exp_folder) if not os.path.isdir(exp_folder) else None

	# save plot
	def save_plot(metric_1, metric_2, name):
		plt.figure()
		plt.plot(model.history.history[metric_1], label=metric_1)
		plt.plot(model.history.history[metric_2], label=metric_2)
		plt.legend(loc='upper left')
		plt.savefig(os.path.join(exp_folder, name))

	save_plot('acc', 'val_acc', 'plot_acc.png')
	save_plot('loss', 'val_loss', 'plot_loss.png')

#####################
### get AUC score ###
#####################

print('\nGetting AUC . . .')

# load best model (fit_generator() returns weights of last epoch)
K.clear_session()
model = load_model(os.path.join(OUTPUT, 'model.h5'), compile=False)
model.compile(
	optimizer=Adam(params['learning_rate']), 
	loss='binary_crossentropy', 
	metrics=['accuracy'])

# reset generator
val_generator = batch_generator(
	train_path,
	f2l_val,
	batch_size,
	mode='eval',
	aug=False)

preds_val = model.predict_generator(
	generator=val_generator, 
	steps=steps_val, 
	verbose=2,
	workers=0)

preds_val = np.squeeze(preds_val)  # save for ensembling

auc_val = roc_auc_score(np.array(f2l_val['label']), preds_val)

########################
### make predictions ###
########################

print('\nMaking predictions . . .')

ts = time.time()

test_files = [f for f in os.listdir(test_path) if f.endswith('.tif')]
if params['debug'] == True:
	test_files = test_files[0:100]

N_test = len(test_files)
steps_test = int(np.ceil(N_test / batch_size))

preds_test = []
idx = 0
for i in range(steps_test):
	imgs = []

	# fill batch
	while len(imgs) < batch_size:
		f = test_files[idx]
		f_path = os.path.join(test_path, f)
		img = load_image(f_path)
		img = preprocess_input(img)
		imgs.append(img)
		idx += 1
		if idx == N_test:
			break

	imgs = np.array(imgs)

	if params['test_time_aug'] == True:
		# vertical flip + horizontal flip + both 
		preds_1 = model.predict(imgs).ravel()
		preds_2 = model.predict(imgs[:, ::-1, :, :]).ravel()
		preds_3 = model.predict(imgs[:, :, ::-1, :]).ravel()
		preds_4 = model.predict(imgs[:, ::-1, ::-1, :]).ravel()
		preds_batch = np.mean([preds_1,preds_2,preds_3,preds_4], axis=0)
		preds_test += preds_batch.tolist()
	else:
		preds_batch = model.predict(imgs).ravel()
		preds_test += preds_batch.tolist()

test_ids = [f[:-4] for f in test_files]
submission = pd.DataFrame({'id': test_ids, 'label': preds_test})

time_pred = np.round((time.time() - ts) / 60, 2)

####################
### export stuff ###
####################

print('\nSaving results . . .')

summary = pd.DataFrame(
	[architecture, auc_val, time_train, time_pred],
	index=['architecture', 'auc_val', 'time_train', 'time_pred'])

if params['debug'] == False:	
	model.save(os.path.join(exp_folder, 'model.h5'))
	params = pd.DataFrame.from_dict(params, orient='index')
	params.to_csv(os.path.join(exp_folder, 'params.csv'), header=False)
	submission.to_csv(os.path.join(exp_folder, 'submission.csv'), index=False)
	summary.to_csv(os.path.join(exp_folder, 'summary.csv'), header=False)

	preds_val = pd.DataFrame(preds_val)
	preds_val.to_csv(
		os.path.join(exp_folder, 'preds_val.csv'), 
		header=False, 
		index=False)
	preds_test = pd.DataFrame(preds_test)
	preds_test.to_csv(
		os.path.join(exp_folder, 'preds_test.csv'), 
		header=False, 
		index=False)

	os.rename(log_file, os.path.join(exp_folder, log_name))  # save log

	# rename exp folder to include AUC score
	new_name = exp_folder + '_score_' + str(round(auc_val, 6))
	os.rename(exp_folder, new_name)

print('\n-----------------------')
print('SUMMARY:')
print('-VAL AUC: %s' % round(auc_val, 6))
print('-TRAIN TIME: %s' % time_train)
print('-PRED TIME: %s' % time_pred)
print('-----------------------')
