import distance
import imagehash
from os import listdir
from os.path import abspath, isfile, join
import pickle
from PIL import Image
from tqdm import tqdm

"""
Data preparation

NOTE:
- This script identifies exact duplicates present in the train set
  (it will not spot near duplicates!).
- Images are hashed using the perceptual hash 'dhash' (difference hash).
- For a nice explanation of the dhash function, see:
  http://tech.jetsetter.com/2017/03/21/duplicate-image-detection/

TODO:
- to find near duplicates, could to get vaguely similar pictures (based on some
  hammer distance threshold of hashes) and then somehow compare their intensity 
  histogram (e.g. look at some quantiles) after different rotations, etc.
"""

# paths
ROOT = abspath('')
DATA = ROOT + '/input/train'  # where pics are located
OUT  = ROOT + '/output/duplicates'  # where duplicates are stored

# get file names
images = [f for f in listdir(DATA) if f.endswith('.tif')]

# get hash
def get_hash(img_path):
	img = Image.open(img_path)
	h = str(imagehash.dhash(img))
	return h

# init hash dict with hash of first image
hashes = {get_hash(join(DATA, images[0])): [images[0]]}

for img in tqdm(images[1:]):
	found_dup = False	
	new_hash = get_hash(join(DATA, img))
	for h in hashes.keys():
		hd = distance.hamming(h, new_hash)
		if hd == 0:
			hashes[h].append(img)
			found_dup = True
			break
	if found_dup == False:
		hashes[new_hash] = [img]

duplicates = {}
for h in hashes.keys():
	if len(hashes[h]) > 1:
		duplicates[h] = hashes[h]

with open(OUT + '/duplicates.pickle', 'wb') as f:
    pickle.dump(duplicates, f)

