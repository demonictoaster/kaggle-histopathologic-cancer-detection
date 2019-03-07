import cv2
import datetime
import json
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.nasnet import NASNetMobile
from keras.applications.nasnet import preprocess_input
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras import backend as K

from config import *

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

TODO:
- data augmentation
- custom callback function
	need to feed val generator and labels to callback fct
	https://github.com/keras-team/keras/issues/3230
"""
K.clear_session()

train_path = os.path.join(INPUT, 'train')
test_path = os.path.join(INPUT, 'test')

OUT  = os.path.join(OUTPUT, 'benchmark')
if not os.path.exists(OUT):
    os.makedirs(OUT)

file_label_map = pd.read_csv(INPUT + '/train_labels.csv')
if params['debug'] == True:
	file_label_map = file_label_map.iloc[0:300]

f2l_train, f2l_val = train_test_split(
	file_label_map, 
	test_size=params['val_size'],
	random_state=12
	)

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
					mode='train', 
					aug=None):

	# shuffle data at each new epoch
	# don't set seed otherwise always same batches
	if mode == 'train':
		f2l = f2l.reindex(np.random.permutation(f2l.index), copy=True)

	n_files = f2l.shape[0]
	idx = 0

	while True:
		imgs = []
		lbls = []

		# fill the batch
		while len(imgs) < batch_size:
			f = f2l.iloc[idx, 0] + '.tif'
			f_path = os.path.join(input_path, f)
			img = load_image(f_path)
			img = preprocess_input(img)
			imgs.append(img)

			lbl = f2l.iloc[idx, 1]
			lbls.append(lbl)

			idx += 1

			if idx == n_files:
				idx = 0  # in case we go through all files
				if mode == 'eval':
					break

		if (aug != None) & (mode=='train'):
			pass

		imgs = np.array(imgs)
		lbls = np.array(lbls).reshape(len(lbls), 1)

		yield(imgs, lbls)

# initialize the generators
train_generator = batch_generator(
	train_path,
	f2l_train,
	batch_size,
	mode='train',
	aug=None)

val_generator = batch_generator(
	train_path,
	f2l_val,
	batch_size,
	mode='eval',
	aug=None)

# checks
# cnt = 0
# for i, j in train_generator:
# 	cnt += 1
# 	print(cnt)
# 	print(i.shape)
# 	print(np.mean(i))
# 	print(k)
# 	print('-------')
# 	if cnt==25:
# 		break

###################
### build model ###
###################

print('\nBuilding model . . .')

# custom callback to print validation AUC at end of epoch
class auc_callback(Callback):
	def __init__(self, val_gen, val_steps, val_labels):
		self.val_gen = val_gen
		self.val_steps = val_steps
		self.labels = val_labels
		self.aucs = []

	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		preds = self.model.predict_generator(self.val_gen, self.val_steps)
		labels = self.labels

		auc = roc_auc_score(labels, preds)
		self.aucs.append(auc)

		print('\rauc_val: %s' % (str(round(auc,4))))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

callback_list = [
	EarlyStopping(
		monitor='val_acc', 
		patience=params['patience']
		),
	ModelCheckpoint(
		filepath=os.path.join(OUT, 'model.h5'), 
		monitor='val_acc', 
		save_best_only=True,
		mode='max'
		),
	auc_callback(
		val_generator, 
		steps_val, 
		f2l_val['label']
		)
	]

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
	X_input = Input(shape=input_shape)
	base_model = NASNetMobile(
		input_tensor=X_input,
		include_top=False, 
		weights='imagenet'
		)

	# if params['debug']==True:
	# 	for l in base_model.layers:
	#     	l.trainable = False

	X = base_model(X_input)
	X = GlobalAveragePooling2D()(X)
	X = Dense(1024, activation='relu')(X)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)

	model.compile(
		optimizer='adam',
		loss='binary_crossentropy',
		metrics=['accuracy']
		)

	return model

model = build_model()
model.summary()

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
    verbose=1
    )

time_train = np.round((time.time() - ts) / 60, 2)

####################################
### evaluate perf validation set ###
####################################

print('\nEvaluating model . . .')

ts = time.time()

# load best model (model.fit returns weights of last training epoch)
K.clear_session()
model = load_model(os.path.join(OUT, 'model.h5'), compile=False)
model.compile(
	optimizer='adam', 
	loss='binary_crossentropy', 
	metrics=['accuracy']
	)

# reset validation generator
val_generator = batch_generator(
	train_path,
	f2l_val,
	batch_size,
	mode='eval',
	aug=None)

preds = model.predict_generator(
	generator=val_generator, 
	steps=steps_val, verbose=1)

auc_val = roc_auc_score(np.array(f2l_val['label']), preds)

time_eval = np.round((time.time() - ts) / 60, 2)

# checks
# model.evaluate_generator(generator=val_generator, steps=steps_val, verbose=1)

# def get_pred(file_path):
# 	img = load_image(file_path)
# 	img = preprocess_input(img)
# 	img = np.expand_dims(img, axis=0)
# 	pred = float(model.predict(img))
# 	return pred

# preds = []
# for id in tqdm(f2l_val['id']):
# 	f_path = os.path.join(train_path, id + '.tif')
# 	pred = get_pred(f_path)
# 	preds.append(pred)

# from sklearn.metrics import accuracy_score
# preds_lab = np.array([1 if x>=0.5 else 0 for x in preds])
# accuracy_score(np.array(f2l_val['label']), preds_lab)

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

preds = []
idx = 0
for i in tqdm(range(steps_test)):
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

	# predict and add to list
	imgs = np.array(imgs)
	pred = model.predict(imgs)
	preds.extend(pred)

preds = np.squeeze(preds)

test_ids = [f[:-4] for f in test_files]
submission = pd.DataFrame({'id': test_ids, 'label': preds})

time_pred = np.round((time.time() - ts) / 60, 2)

####################
### export stuff ###
####################

print('\nSaving results . . .')

times = [time_train, time_eval, time_pred]

def export_model(out_folder, score, model, params, submission, times):
	score = round(score, 6)
	today = datetime.datetime.now()
	folder_name = today.strftime('%y%m%d') + '_' + today.strftime("%H%M") + \
 		'_score_' + str(score)
	folder = os.path.join(out_folder, folder_name)
	os.mkdir(folder)

	model.save(os.path.join(folder, 'model.h5'))

	with open(os.path.join(folder, 'params.json'), 'w') as f:
		json.dump(params, f)

	submission.to_csv(os.path.join(folder, 'submission.csv'), index=False)

	time_file = os.path.join(folder, 'compute_time.csv')
	with open(time_file, 'w') as f:
		f.write('train_time: %s\n' % times[0])
		f.write('eval_time: %s\n' % times[1])
		f.write('pred_time: %s\n' % times[2])

if params['debug'] == False:
	export_model(OUT, auc_val, model, params, submission, times)


print('\n-----------------------')
print('SUMMARY:')
print('-VAL AUC: %s' % round(auc_val, 6))
print('-TRAIN TIME: %s' % time_train)
print('-EVAL TIME: %s' % time_eval)
print('-PRED TIME: %s' % time_pred)
print('-----------------------')
