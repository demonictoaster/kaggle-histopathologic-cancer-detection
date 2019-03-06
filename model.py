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
NOTE:
- freezing InceptionV3 layers leads to weird results at test time due to
  bug linked to batch normalization layers 
  (http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/)

TODO:
- data augmentation
- custom callback function
- fix val acc constant across epochs
- speed up eval and pred parts (pred by batch)
"""
K.clear_session()

np.random.seed(12)

train_path = os.path.join(INPUT, 'train')
test_path = os.path.join(INPUT, 'test')

OUT  = os.path.join(OUTPUT, 'benchmark')
if not os.path.exists(OUT):
    os.makedirs(OUT)

file_label_map = pd.read_csv(INPUT + '/train_labels.csv')
if params['debug'] == True:
	file_label_map = file_label_map.iloc[0:512]

f2l_train, f2l_val = train_test_split(
	file_label_map, 
	test_size=params['val_size'],
	random_state=12
	)

batch_size = params['batch_size']
epochs = params['epochs']
N_train = f2l_train.shape[0]
N_val = f2l_val.shape[0]
steps_train = N_train // batch_size
steps_val = N_val // batch_size

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

	n_files = f2l.shape[0]
	idx = 0
	while True:
		imgs = []
		lbls = []

		# shuffle data at each new epoch
		if mode == 'train':
			f2l = f2l.reindex(np.random.permutation(f2l.index))

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
				idx = 0  # fill batch with first files once reached the end
				if mode == 'eval':
					break

		if aug != None:
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

# test generator
# idx = 0
# for i, j in train_generator:
# 	idx += 1
# 	print(idx)
# 	print(i.shape)
# 	print(np.mean(i))
# 	if idx==25:
# 		break


###################
### build model ###
###################

print('\nBuilding model . . .')

class auc_callback(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
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

time_train = str(np.round((time.time() - ts) / 60, 2))

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

def predict_label(file_path):
	img = load_image(file_path)
	img = preprocess_input(img)
	img = np.expand_dims(img, axis=0)
	pred = float(model.predict(img))
	return pred

preds = []
for id in tqdm(f2l_val['id']):
	f_path = os.path.join(train_path, id + '.tif')
	pred = predict_label(f_path)
	preds.append(pred)

auc_val = roc_auc_score(np.array(f2l_val['label']), preds)

time_eval = str(np.round((time.time() - ts) / 60, 2))

model.evaluate_generator(generator=val_generator, steps=steps_val, verbose=1)
preds = model.predict_generator(generator=val_generator, steps=steps_val, verbose=1)
preds = np.squeeze(preds)

########################
### make predictions ###
########################

print('\nMaking predictions . . .')

ts = time.time()

test_files = [f for f in os.listdir(test_path) if f.endswith('.tif')]
if params['debug'] == True:
	test_files = test_files[0:100]

preds = []
for f in tqdm(test_files):
	f_path = os.path.join(test_path, f)
	pred = predict_label(f_path)
	preds.append(pred)	

time_pred = str(np.round((time.time() - ts) / 60, 2))

test_ids = [f[:-4] for f in test_files]
submission = pd.DataFrame({'id': test_ids, 'label': preds})

####################
### export stuff ###
####################

print('\nSaving results . . .')

def export_model(out_folder, score, model, params, submission):
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


if params['debug'] == False:
	export_model(OUT, auc_val, model, params, submission)


print('\n-----------------------')
print('SUMMARY:')
print('-VAL AUC: %s' % round(auc_val, 6))
print('-TRAIN TIME: %s' % time_train)
print('-EVAL TIME: %s' % time_eval)
print('-PRED TIME: %s' % time_pred)
print('-----------------------')
