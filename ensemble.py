import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# load data
ROOT = os.path.abspath('')
ensemble_path = os.path.join(ROOT, 'ensemble')

example = pd.read_csv(os.path.join(ensemble_path, 'example_submission.csv'))

data = {}
for f in os.listdir(ensemble_path):
	file_path = os.path.join(ensemble_path, f)
	if os.path.isfile(file_path):
		data[f[:-4]] = pd.read_csv(file_path, header=None)


# simple averaging of predictions
d1 = data['preds_test_inception'].values.ravel()
d2 = data['preds_test_nasnet'].values.ravel()
d3 = data['preds_test_densenet'].values.ravel()

preds = np.mean([d1, d2, d3], axis=0)

submission = pd.DataFrame({'id': example['id'], 'label': preds})
submission.to_csv(
	os.path.join(ensemble_path, 'output/submission_average.csv'), 
	index=False)

# linear combination
preds = 0.4 * d1 + 0.3 * d2 + 0.3 * d3

submission = pd.DataFrame({'id': example['id'], 'label': preds})
submission.to_csv(
	os.path.join(ensemble_path, 'output/submission_linear.csv'), 
	index=False)

