#!/usr/bin/env python
# coding: utf-8

# In[ ]:




from os.path import splitext, join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import h5py
import cv2

class BasicDataset(Dataset):
    def __init__(self, channels, mode='', input='', tag=''):
        dir= '/content/drive/MyDrive/BIA2022/data'
        if mode == 'sampled':
            self.niom = h5py.File(join(dir, 'niom_sampled_train.h5'), 'r')
            self.mem = h5py.File(join(dir, 'mem_sampled_train.h5'), 'r')
        if mode == 'full':
            self.niom = h5py.File(join(dir, 'niom_full_train.h5'), 'r')
            self.mem = h5py.File(join(dir, 'mem_full_train.h5'), 'r')
        self.input = input
        self.channels = channels
        self.slices = []
        self.tag = tag
        for f in sorted(list(self.mem.keys())):
            for iz in range(self.channels,self.mem[f].shape[0]):
                self.slices += [(f, iz)] 
        
        self.ids = self.slices
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        key, slice_id = self.slices[i]

        n_size = 64
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mem_slice = np.zeros(((self.channels,n_size,n_size)))
        for j in range(self.channels):
            mem_slice[j,:,:] = cv2.resize(self.mem[key][slice_id-self.channels+j,:,:],(n_size,n_size),interpolation=cv2.INTER_CUBIC)
            if self.input == 'mask':
                mem_slice[j,:,:] = cv2.GaussianBlur(mem_slice[j,:,:], (7, 7), 0)
                mem_slice[j,:,:] = cv2.normalize(mem_slice[j,:,:], np.zeros(((1,n_size,n_size))), 0, 255, cv2.NORM_MINMAX)
                ret, mem_slice[j,:,:] = cv2.threshold(mem_slice[j,:,:].astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
                mem_slice[j,:,:] = cv2.normalize(mem_slice[j,:,:], np.zeros(((1,n_size,n_size))), 0, 1, cv2.NORM_MINMAX)
            if self.input == 'image':
                mem_slice[j,:,:] = cv2.normalize(mem_slice[j,:,:],  np.zeros(((1,n_size,n_size))), 0, 255, cv2.NORM_MINMAX)


        niom_slice = np.zeros(((1, n_size, n_size)))
        niom_slice[0, :, :] = cv2.resize(self.niom[key][slice_id, :, :], (n_size, n_size),
                                         interpolation=cv2.INTER_CUBIC)
        niom_slice[0, :, :] = cv2.GaussianBlur(niom_slice[0, :, :], (7, 7), 0)
        niom_slice[0, :, :] = cv2.normalize(niom_slice[0, :, :], np.zeros(((1,n_size,n_size))), 0, 255, cv2.NORM_MINMAX)
        ret, niom_slice[0, :, :] = cv2.threshold(niom_slice[0, :, :].astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        niom_slice[0, :, :] = cv2.GaussianBlur(niom_slice[0, :, :], (7, 7), 0)
        niom_slice[0, :, :] = cv2.erode(niom_slice[0, :, :], kernel, iterations=1)
        niom_slice[0, :, :] = cv2.GaussianBlur(niom_slice[0, :, :], (7, 7), 0)
        niom_slice[0, :, :] = cv2.normalize(niom_slice[0, :, :], np.zeros(((1,n_size,n_size))), 0, 1, cv2.NORM_MINMAX)

        if self.tag == 'ConvLSTM':
            mem_slice = mem_slice[..., np.newaxis]
            mem_slice = torch.tensor(mem_slice).permute(0, 3, 1, 2).contiguous()
        return {
           'image': mem_slice,
           'mask': torch.tensor(niom_slice)
        }

