import os

STORAGE = '/storage/kaggle_cancer_competition/temp'
# STORAGE = '/Users/mike/Dropbox/computer_stuff/ML/kaggle/histopathologic_cancer_detection/temp'

print('\n pics in train 0: ')
files = [f for f in os.listdir(os.path.join(STORAGE, 'train', '0')) if f.endswith('.tif')]
print(len(files))

print('\n pics in train 1: ')
files = [f for f in os.listdir(os.path.join(STORAGE, 'train', '1')) if f.endswith('.tif')]
print(len(files))

print('\n pics in val 0: ')
files = [f for f in os.listdir(os.path.join(STORAGE, 'val', '0')) if f.endswith('.tif')]
print(len(files))

print('\n pics in val 1: ')
files = [f for f in os.listdir(os.path.join(STORAGE, 'val', '1')) if f.endswith('.tif')]
print(len(files))