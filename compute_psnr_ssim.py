import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from dipy.io.image import load_nifti, save_nifti
import numpy as np
import torch
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.image import load_nifti, save_nifti
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# load data
ground_truth,affine=load_nifti("")
simulated,affine=load_nifti("")

# normalize data
ground_truth = (ground_truth.astype(np.float32)-ground_truth.min())/(ground_truth.max()-ground_truth.min())
simulated = (simulated.astype(np.float32)-simulated.min())/(simulated.max()-simulated.min())


psnr=np.zeros(64)
ssim=np.zeros(64)

psnrsum=0
ssimsum=0

for i in range(64):
    for j in range(72):

        psnrij = peak_signal_noise_ratio(ground_truth[:,:,j,:].squeeze(2), simulated[:,:,j,i],data_range=1)
        ssimij = structural_similarity(ground_truth[:,:,j,:].squeeze(2), simulated[:,:,j,i],data_range=1)
        psnrsum = psnrsum+psnrij
        ssimsum = ssimsum+ssimij

    psnr[i]=psnrsum/72
    ssim[i]=ssimsum/72
    psnrsum=0
    ssimsum=0

mean1 = np.mean(psnr)
mean2 = np.mean(ssim)
std1 = np.std(psnr)
std2 = np.std(ssim)

print("psnr:")
print("Mean:", mean1)
print("Standard Deviation:", std1)# not used now

print("ssim:")
print("Mean:", mean2*100)
print("Standard Deviation:", std2)# not used now