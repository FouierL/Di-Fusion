from curses import raw
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torch
from dipy.io.image import save_nifti, load_nifti
from matplotlib import pyplot as plt
from torchvision import transforms, utils


class MRIDataset(Dataset):
    def __init__(self, dataroot, valid_mask, phase='train', image_size=128, in_channel=1, val_volume_idx=50, val_slice_idx=40,
                 padding=1, lr_flip=0.5, stage2_file=None, PAE_dataroot= None,noise2noise=None,gt=None):
        self.padding = padding // 2
        self.lr_flip = lr_flip
        self.phase = phase
        self.in_channel = in_channel
        self.PAE_dataroot = PAE_dataroot
        self.noise2noise = noise2noise
        self.gt_e = gt
        # read data
        raw_data, _ = load_nifti(dataroot) # width, height, slices, gradients
        
        print('Loaded data of size:', raw_data.shape)
        
        # normalize data
        raw_data = raw_data.astype(np.float32) / np.max(raw_data, axis=(0,1,2), keepdims=True)
        
        # parse mask
        assert type(valid_mask) is (list or tuple) and len(valid_mask) == 2
 
        # mask data
        raw_data = raw_data[:,:,:,valid_mask[0]:valid_mask[1]]
        
        print('Used data of size:', raw_data.shape) 
        self.data_size_before_padding = raw_data.shape

        self.raw_data = np.pad(raw_data, ((0,0), (0,0), (in_channel//2, in_channel//2), (self.padding, self.padding)), mode='wrap')
        

        if gt is not None:
            gt, _ = load_nifti(gt)
            print('Loaded gt of size:', gt.shape)
            gt = gt[:,:,:,(valid_mask[0]):(valid_mask[1])] 
            print('Used gt of size:', gt.shape)
            gt = gt.astype(np.float32) / np.max(gt, axis=(0,1,2), keepdims=True)
            self.gt = np.pad(gt, ((0,0), (0,0), (0,0), (0,0)), mode='wrap')

        # running for Stage3?
        if stage2_file is not None:
            print('Parsing Stage2 matched states from the stage2 file...')
            self.matched_state = self.parse_stage2_file(stage2_file)
        else:
            self.matched_state = None

        # transform
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Resize(image_size),
                transforms.RandomVerticalFlip(lr_flip),
                transforms.RandomHorizontalFlip(lr_flip),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Resize(image_size),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

        # prepare validation data
        if val_volume_idx == 'all':
            self.val_volume_idx = range(raw_data.shape[-1])
        elif type(val_volume_idx) is int:
            self.val_volume_idx = [val_volume_idx]
        elif type(val_volume_idx) is list:
            self.val_volume_idx = val_volume_idx
        else:
            self.val_volume_idx = [int(val_volume_idx)]

        if val_slice_idx == 'all':
            self.val_slice_idx = range(0, raw_data.shape[-2])
        elif type(val_slice_idx) is int:
            self.val_slice_idx = [val_slice_idx]
        elif type(val_slice_idx) is list:
            self.val_slice_idx = val_slice_idx
        else:
            self.val_slice_idx = [int(val_slice_idx)]

    def parse_stage2_file(self, file_path):
        results = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                info = line.strip().split('_')
                volume_idx, slice_idx, t = int(info[0]), int(info[1]), int(info[2])
                if volume_idx not in results:
                    results[volume_idx] = {}
                results[volume_idx][slice_idx] = t
        return results
    
    def _add_noise(self, img):
        """Adds Gaussian noise to image."""
        std = np.random.uniform(0, 0.5)
        # std = 0.5
        noise = np.random.normal(0, std, img.shape)
        noise = torch.tensor(noise).to(img.device).to(torch.float32)
        return img+noise

    def __len__(self):
        if self.phase == 'train' or self.phase == 'test':
            return self.data_size_before_padding[-2] * self.data_size_before_padding[-1] # num of volumes
        elif self.phase == 'val':
            return len(self.val_volume_idx) * len(self.val_slice_idx)

    def __getitem__(self, index):
        if self.phase == 'train' or self.phase == 'test':
            # decode index to get slice idx and volume idx
            volume_idx = index // self.data_size_before_padding[-2]
            slice_idx = index % self.data_size_before_padding[-2]
        elif self.phase == 'val':
            s_index = index % len(self.val_slice_idx)
            index = index // len(self.val_slice_idx)
            slice_idx = self.val_slice_idx[s_index]
            volume_idx = self.val_volume_idx[index]

        raw_input = self.raw_data
        
            
        if self.padding > 0:
            if self.gt_e is not None:
                gt = self.gt
                raw_input = np.concatenate((
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,volume_idx:volume_idx+self.padding],
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,volume_idx+self.padding+1:volume_idx+2*self.padding+1],
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]), axis=-1)
            else :
                raw_input = np.concatenate((
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,volume_idx:volume_idx+self.padding],
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,volume_idx+self.padding+1:volume_idx+2*self.padding+1],
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]), axis=-1)

        elif self.padding == 0:
            raw_input = np.concatenate((
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding-1]],
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]), axis=-1)
        
        if len(raw_input.shape) == 4:
            raw_input = raw_input[:,:,0]
        

        raw_input = self.transforms(raw_input) # only support the first channel for now
        if self.noise2noise:
            ret = dict(X=self._add_noise(raw_input[[-1], :, :].to(torch.float32)), condition=raw_input[:-1, :, :].to(torch.float32))
            ret['target']=raw_input[[-1], :, :].to(torch.float32)
        elif self.gt_e:
            ret = dict(X=(raw_input[[-1], :, :].to(torch.float32)), condition=raw_input[:-1, :, :].to(torch.float32))
            ret['target']=raw_input[[-1], :, :].to(torch.float32)
        else:
            ret = dict(X=raw_input[[-1], :, :].to(torch.float32), condition=raw_input[:-1, :, :].to(torch.float32))
        ret['slice_idx']=slice_idx
        for i in range(0,self.padding*2):
            ret['condition'+str(i)]=raw_input[i,:,:].unsqueeze(0).to(torch.float32)
        

        return ret


if __name__ == "__main__":

    print("im ok")

