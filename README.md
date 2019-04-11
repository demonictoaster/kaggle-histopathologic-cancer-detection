## Kaggle Histopathologic Cancer Detection Competition
Code used for competition submission. This solution got me in the top 14% on the private leaderboard (out of 1157 competitors). Unfortunately there was a data leak near the end of the competition. Exploiting the leak, some participants were able to get a perfect score.

# Overview
The goal is to detect the presence of metastatic cancer in image patches taken from pathology scans (i.e. binary classification task).

# Approach
I tried different CNN architectures and trained with weights initialized based on ImageNet. The main steps leading to performance gain were:
* data augmentation (using the `imgaug` library)
* test time augmentation
* ensembling

Training took about 4-6 hours (depending on CNN architecture) on a P4000 GPU for 12 epochs. Increasing the number of epochs might lead to performance improvements (although I saw signs of overfitting after about 9-10 epochs already).

The final ensemble yielded a 0.9696 AUC on the private leaderboard. 

# Dependencies
* `tensorflow 1.5.0`
* `keras 2.2.4`
* `opencv-python`
* `imgaug`
* `sklearn`
* `pandas`
