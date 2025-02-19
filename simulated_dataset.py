import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
import fastmri
from fastmri.data import transforms as T
import torch
from dipy.io.image import load_nifti, save_nifti
import numpy as np
import torch
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

data_folder = ''; # please change this to your data folder
noise_level= 7 #please specify noise_level here
dwi_fname, dwi_bval_fname, dwi_bvec_fname= get_fnames('sherbrooke_3shell')
data, affine = load_nifti(dwi_fname)

# select data from fastMRI dataset
file_name1 = os.path.join(data_folder, '2022061203_T101.h5')
hf1 = h5py.File(file_name1)

file_name2 = os.path.join(data_folder, '2022061204_T101.h5')
hf2 = h5py.File(file_name1)

file_name3 = os.path.join(data_folder, '2022061205_T101.h5')
hf3 = h5py.File(file_name1)

file_name4 = os.path.join(data_folder, '2022061206_T101.h5')
hf4 = h5py.File(file_name1)

print('Keys:', list(hf1.keys()))
print('Attrs:', dict(hf1.attrs))

volume_kspace1 = hf1['kspace'][()]
print('hf1 Kspace shape:',volume_kspace1.shape)
out1 = hf1['reconstruction_rss'][()]
print('hf1 reconstruction rss shape:',out1.shape)

volume_kspace2 = hf2['kspace'][()]
print('hf2 Kspace shape:',volume_kspace2.shape)
out2 = hf2['reconstruction_rss'][()]
print('hf2 reconstruction rss shape:',out2.shape)

volume_kspace3 = hf3['kspace'][()]
print('hf3 Kspace shape:',volume_kspace3.shape)
out3 = hf3['reconstruction_rss'][()]
print('hf3 reconstruction rss shape:',out3.shape)

volume_kspace4 = hf4['kspace'][()]
print('hf4 Kspace shape:',volume_kspace4.shape)
out4 = hf4['reconstruction_rss'][()]
print('hf4 reconstruction rss shape:',out4.shape)

ground_truth=torch.zeros(1,72,256,256)
volume_pic=torch.zeros(64,72,256,256)

for i in range(volume_kspace1.shape[0]):
    slice_kspace1 = T.to_tensor(volume_kspace1[i])
    slice_kspace2 = T.to_tensor(volume_kspace2[i])
    slice_kspace3 = T.to_tensor(volume_kspace3[i])
    slice_kspace4 = T.to_tensor(volume_kspace4[i])

    slice_image1 = fastmri.ifft2c(slice_kspace1)
    slice_image_abs1 = fastmri.complex_abs(slice_image1)

    slice_image2 = fastmri.ifft2c(slice_kspace2)
    slice_image_abs2 = fastmri.complex_abs(slice_image2)

    slice_image3 = fastmri.ifft2c(slice_kspace3)
    slice_image_abs3 = fastmri.complex_abs(slice_image3)

    slice_image4 = fastmri.ifft2c(slice_kspace4)
    slice_image_abs4 = fastmri.complex_abs(slice_image4)

    slice_image_rss1 = fastmri.rss(slice_image_abs1, dim=0)
    slice_image_rss2 = fastmri.rss(slice_image_abs2, dim=0)
    slice_image_rss3 = fastmri.rss(slice_image_abs3, dim=0)
    slice_image_rss4 = fastmri.rss(slice_image_abs4, dim=0)
    
    
    ground_truth[:,0+i,:,:]=slice_image_rss1
    ground_truth[:,18+i,:,:]=slice_image_rss2
    ground_truth[:,36+i,:,:]=slice_image_rss3
    ground_truth[:,54+i,:,:]=slice_image_rss4

ground_truth=ground_truth.numpy()
print(ground_truth.shape)
save_nifti('ground_truth.nii.gz', ground_truth, affine)


for i in range(volume_kspace1.shape[0]):
    slice_kspace1 = T.to_tensor(volume_kspace1[i])
    slice_kspace2 = T.to_tensor(volume_kspace2[i])
    slice_kspace3 = T.to_tensor(volume_kspace3[i])
    slice_kspace4 = T.to_tensor(volume_kspace4[i])

    
    for k in range(64):
        
        noise_real=torch.randn_like(slice_kspace1[:,:,:,0])
        noise_imaginary=torch.randn_like(slice_kspace1[:,:,:,1])
        
        noise_real=noise_real*noise_level
        noise_imaginary=noise_imaginary*noise_level
        
        slice_ksapce1_noisy=torch.cat([(slice_kspace1[:,:,:,0]+noise_real).unsqueeze(-1),(slice_kspace1[:,:,:,1]+noise_imaginary).unsqueeze(-1)],dim=-1)
        slice_ksapce2_noisy=torch.cat([(slice_kspace2[:,:,:,0]+noise_real).unsqueeze(-1),(slice_kspace2[:,:,:,1]+noise_imaginary).unsqueeze(-1)],dim=-1)
        slice_ksapce3_noisy=torch.cat([(slice_kspace3[:,:,:,0]+noise_real).unsqueeze(-1),(slice_kspace3[:,:,:,1]+noise_imaginary).unsqueeze(-1)],dim=-1)
        slice_ksapce4_noisy=torch.cat([(slice_kspace4[:,:,:,0]+noise_real).unsqueeze(-1),(slice_kspace4[:,:,:,1]+noise_imaginary).unsqueeze(-1)],dim=-1)

        slice_image1 = fastmri.ifft2c(slice_ksapce1_noisy)
        slice_image_abs1 = fastmri.complex_abs(slice_image1)

        slice_image2 = fastmri.ifft2c(slice_ksapce2_noisy)
        slice_image_abs2 = fastmri.complex_abs(slice_image2)

        slice_image3 = fastmri.ifft2c(slice_ksapce3_noisy)
        slice_image_abs3 = fastmri.complex_abs(slice_image3)

        slice_image4 = fastmri.ifft2c(slice_ksapce4_noisy)
        slice_image_abs4 = fastmri.complex_abs(slice_image4)

        slice_image_rss1 = fastmri.rss(slice_image_abs1, dim=0)
        slice_image_rss2 = fastmri.rss(slice_image_abs2, dim=0)
        slice_image_rss3 = fastmri.rss(slice_image_abs3, dim=0)
        slice_image_rss4 = fastmri.rss(slice_image_abs4, dim=0)

        volume_pic[k,0+i,:,:]=slice_image_rss1
        volume_pic[k,18+i,:,:]=slice_image_rss2
        volume_pic[k,36+i,:,:]=slice_image_rss3
        volume_pic[k,54+i,:,:]=slice_image_rss4

volume_pic=volume_pic.numpy()
shuffled_indices = np.random.permutation(64)
volume_pic = volume_pic[shuffled_indices]
print(volume_pic.shape)
save_nifti('simulated.nii.gz', volume_pic, affine)

ground_truth,affine=load_nifti("")
simulated,affine=load_nifti("")

ground_truth = (ground_truth.astype(np.float32)-ground_truth.min())/(ground_truth.max()-ground_truth.min())
simulated = (simulated.astype(np.float32)-simulated.min())/(simulated.max()-simulated.min())
psnrsum=0
ssimsum=0
for i in range(64):
    for j in range(72):

        psnr = peak_signal_noise_ratio(ground_truth[:,j,:,:].squeeze(0), simulated[i,j,:,:],data_range=1)
        ssim = structural_similarity(ground_truth[:,j,:,:].squeeze(0), simulated[i,j,:,:],data_range=1)
        psnrsum=psnrsum+psnr
        ssimsum=ssimsum+ssim

print("PSNR:", psnrsum/(64*72))
print("SSIM:", ssimsum/(64*72))