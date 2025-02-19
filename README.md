# Self-Supervised Diffusion MRI Denoising via Iterative and Stable Refinement (ICLR 2025)
## 📖[**Paper**](https://arxiv.org/abs/2501.13514)

[Chenxu Wu](https://fouierl.github.io/chenxuwu.github.io/), [Qingpeng Kong](https://kqp1227.github.io/), [Zihang Jiang](https://scholar.google.com/citations?user=Wo8tMSMAAAAJ), [S.Kevin Zhou](https://scholar.google.com/citations?user=8eNm2GMAAAAJ)
from [MIRACLE](https://miracle.ustc.edu.cn/main.htm), USTC

!!! important !!!
This repository is highly built on https://github.com/StanfordMIMI/DDM2 and https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement?tab=Apache-2.0-1-ov-file
Thanks for open-sourcing
*Please respect their license of usage.*
!!! important !!!

## Dependencies

Please clone the environment using the following command:

```
conda env create -f environment.yml  
conda activate difusion
```

## Quick Start

### Datasets

For fair evaluations, we used the data provided in the [DIPY](https://dipy.org/) library. One can easily access their provided data (e.g. Sherbrooke and Stanford HARDI) by using their official loading script:  

```python3
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)
```

For PPMI datasets, one can access them on https://www.ppmi-info.org/
Please note that the data downloaded from PPMI (Parkinson's Progression Markers Initiative) is stored in a format different from nii.gz (NIfTI format). One need to perform the conversion from .nrrd to nii.gz.

For fast MRI datasets, one can access them following the instruction in https://github.com/facebookresearch/fastMRI
Please note that the data downloaded from fastMRI is stored in a format different from nii.gz (NIfTI format). One need to perform the conversion from H5 to nii.gz, details are in ```simulated_dataset.py```.

For making simulated datasets, we use the API in fastMRI and follow the instruction in M4Raw, which are in ```simulated_dataset.py```. 

*This code is derived from this [fastMRI,M4Raw](https://github.com/facebookresearch/fastMRI and https://github.com/mylyu/M4Raw). Please respect their license of usage.*

### Configs

Different experiments are controlled by configuration files, which are in ```config/```. 

We have provided default training configurations for reproducing our experiments. Users are required to **change the path vairables** to their own directory/data before running any experiments.

### Train

For each dataset, a corresponding config file (or an update of the original config file) need to be passed as a command line arg.

 To train our model:  
  ```python3 train_diff_model.py -p train -c config/hardi_150.json```  
  or alternatively, modify ```run_ourmodel.sh``` and run:  
  ```./run_ourmodel.sh```  

Training in the latter diffusion steps:
 line 486 in ```model/mri_modules/diffusion.py```

### Inference (Denoise)

One can use the trained model to denoise a dMRI dataset through:  
```python denoise.py -c config/hardi_150.json```  
or alternatively, modify ```denoise.sh``` and run:  
```./denoise.sh```   

The ```--save``` flag can be used to save the denoised reusults into a single '.nii.gz' file:  
```python denoise.py -c config/hardi_150.json --save```

CSNR:
 line 428 in ```model/mri_modules/diffusion.py```

## Reproduce The Results In The Paper

### Our Denoised Results and Simulated Datasets

The denoised results and simulated datasets can be found at [data_share](https://www.jianguoyun.com/p/DYCPcQkQgOSNDBjptPcFIAA), for Quantitative Metrics Calulation, use hardi150_300t_0.85.nii.gz, for other tasks, use hardi150_300t_0.40.nii.gz.

### Quantitative Metrics Calulation

With the denoised dataset, please follow the instructions in ```quantitative_metrics.ipynb``` in https://github.com/StanfordMIMI/DDM2 to calculate SNR and CNR scores.

*The notebook is derived from this [DIPY script](https://docs.dipy.org/stable/examples_built/preprocessing/snr_in_cc.html#sphx-glr-examples-built-preprocessing-snr-in-cc-py). Please respect their license of usage.*

### Tractography

With the denoised dataset, please follow the instructions in ```tractographyTracking_FiberBundleCoherency.ipynb``` in https://github.com/ShreyasFadnavis/patch2self to perform tractography.

*The notebook is derived from this [DIPY script](https://docs.dipy.org/stable/examples_built/contextual_enhancement/fiber_to_bundle_coherence.html#sphx-glr-examples-built-contextual-enhancement-fiber-to-bundle-coherence-py). Please respect their license of usage.*

### Microstructure model fitting

With the denoised dataset, please follow the instructions in ```voxel_k-fold_crossvalidation.ipynb``` in https://github.com/ShreyasFadnavis/patch2self to perform microstructure model fitting.

*The notebook is derived from this [DIPY script](https://docs.dipy.org/stable/examples_built/reconstruction/kfold_xval.html#sphx-glr-examples-built-reconstruction-kfold-xval-py). Please respect their license of usage.*

### Reconstruction of the diffusion signal

With the denoised dataset, please follow the instructions in ```reconst_dki.ipynb``` in https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html#sphx-glr-examples-built-reconstruction-reconst-dki-py to perform the reconstruction of the diffusion signal.

*The notebook is derived from this [DIPY script](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html#sphx-glr-examples-built-reconstruction-reconst-dki-py). Please respect their license of usage.*

### Simulated experiment SSIM and PSNR

With the denoised dataset, please follow the instructions in ```compute_psnr_ssim.py.py```.


# References
If you find this repository useful for your research, please cite the following work.
```
@inproceedings{wuself,
  title={Self-Supervised Diffusion MRI Denoising via Iterative and Stable Refinement},
  author={Wu, Chenxu and Kong, Qingpeng and Jiang, Zihang and Zhou, S Kevin},
  booktitle={The Thirteenth International Conference on Learning Representations}
  year={2025}
}
```
This implementation is based on / inspired by:
- https://github.com/StanfordMIMI/DDM2 (DDM2)
- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement?tab=Apache-2.0-1-ov-file (SR3)
Thanks again for open-sourcing!