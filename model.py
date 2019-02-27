import datetime
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
# from tensorflow.keras.applications.nasnet import NASNetMobile
# from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

from config import *

"""
NOTE:
- freezing InceptionV3 layers leads to weird results at test time due to
  bug linked to batch normalization layers 
  (see http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/)

TODO:
- 
"""

K.clear_session()
OUT  = os.path.join(OUTPUT, 'benchmark')
if not os.path.exists(OUT):
    os.makedirs(OUT)

batch_size = params['batch_size']
epochs = params['epochs']
steps_train = int(N_train / batch_size)
steps_val   = N_val

#################
### daga prep ###
#################

datagen = image.ImageDataGenerator(
	preprocessing_function = preprocess_input
	)

train_generator = datagen.flow_from_directory(
	TRAIN,
	target_size=(IMG_SIZE, IMG_SIZE),
	batch_size=batch_size,
	class_mode='binary'	)

val_generator = datagen.flow_from_directory(
	VAL,
	target_size=(IMG_SIZE, IMG_SIZE),
	batch_size=1,
	class_mode='binary',
	shuffle=False
	)

#######################
### benchmark model ###
#######################

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
	X_input = Input(input_shape)
	base_model = InceptionV3(
		weights='imagenet', 
		include_top=False, 
		input_shape=input_shape
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

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_train,
    epochs=epochs,
    callbacks=callback_list,
    validation_data=val_generator,
    validation_steps=steps_val,
    verbose=1
    )

# load best model (model.fit returns weights of last training epoch)
K.clear_session()
model = load_model(os.path.join(OUT, 'model.h5'))

####################################
### evaluate perf validation set ###
####################################

y_val_pred = model.predict_generator(
	generator=val_generator,
	steps=steps_val,
	verbose=1
	)

auc_val = roc_auc_score(val_generator.classes, y_val_pred)
auc_val


# checks
from sklearn.metrics import accuracy_score
y_pred = np.array([1 if x>=0.5 else 0 for x in y_val_pred])
accuracy_score(val_generator.classes, y_pred)

# model.evaluate_generator(generator=val_generator, steps=steps_val, verbose=1)


########################
### make predictions ###
########################

test_ids = [f for f in os.listdir(TEST) if f.endswith('.tif')]
preds = []

for id in tqdm(test_ids):
	img_path = os.path.join(TEST, id)
	img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	pred = float(model.predict(x))
	preds.append(pred)

submission = pd.DataFrame({'id': test_ids, 'label': preds})

####################
### export stuff ###
####################

def export_model(out_folder, score, model, params, submission):
	
	# create folder
	score = round(score, 6)
	today = datetime.datetime.now()
	sub_id = today.strftime('%y%m%d') + '_' + today.strftime("%H%M") + \
 		'_score_' + str(score)
	folder = out_folder + '/' + sub_id
	os.mkdir(folder)
	print('\n---- ' + sub_id + ' ----')

	# save stuff
	model.save(os.path.join(folder, 'model.h5'))
	with open(os.path.join(folder, 'params.json'), 'w') as f:
    	json.dump(params, f)
	submission.to_csv(os.path.join(folder, 'submission.csv'), index=False)

if params['debug'] == False:
	export_model(OUT, auc_val, model, params, submission)
