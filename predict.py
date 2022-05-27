#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import logging
import os
from os.path import splitext, join
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet, ConvLSTM
from dataset import BasicDataset
import h5py
import cv2

class Config:
    gpus = [2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 17, 16, 3, 1, 1),
               ('conv', 'sigmoid', 16, 1, 1, 0, 1)]

dir_checkpoint = '/content/drive/MyDrive/BIA2022/checkpoints/'
dir_default = '/content/drive/MyDrive/BIA2022/data'
file_test_sampled = 'mem_sampled_test.h5'
file_test_full = 'mem_full_test.h5'
n_size = 64
def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = full_img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', dest='net', type=str, default='Unet',
                        help='Unet or ConvLSTM')
    parser.add_argument('--input', '-i', metavar='INPUT', type=str,
                        help='image or mask')
    parser.add_argument('-c', '--channels', metavar='C', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)
    parser.add_argument('-mode', '--mode', dest='mode', type=str, default='sampled',
                        help='Sampled - each 25 ms, Full - each slice')
    parser.add_argument('-n','--name', dest='name', type=str, default='experiment_name',
                        help='name of the experiment')
    return parser.parse_args()

def get_output_filenames(args):
    if args.mode == 'sampled':
        in_files = h5py.File(join(dir_default,file_test_sampled), 'r')
    if args.mode == 'full':
        in_files = h5py.File(join(dir_default,file_test_full), 'r')
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

if __name__ == "__main__":
    args = get_args()

    if args.mode == 'sampled':
        in_files = h5py.File(join(dir_default,file_test_sampled), 'r')
        epoch_number = 5
    if args.mode == 'full':
        in_files = h5py.File(join(dir_default,file_test_full), 'r')
        epoch_number = 5
    slices = []
    for f in sorted(list(in_files.keys())):
        for iz in range(args.channels,in_files[f].shape[0]):
            slices += [(f, iz)]

    if args.net == 'Unet':
        net = UNet(in_channels=args.channels, out_channels=1, bilinear=True)
    if args.net == 'ConvLSTM':
        config = Config()
        net = ConvLSTM(config, in_channels=args.channels, out_channels=1).to(config.device)

    print("Loading model {}".format(dir_checkpoint + f'{args.name}_epoch{epoch_number}.pth'))
    logging.info("Loading model {}".format(dir_checkpoint + f'{args.name}_epoch{epoch_number}.pth'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(format(dir_checkpoint + f'{args.name}_epoch{epoch_number}.pth'), map_location=device))

    print("Model loaded !")
    
    out_files = np.zeros(((in_files[f].shape[0],n_size,n_size)))
    
    for i, fn in enumerate(slices):
        if i % (100) == 0:
            print("\nPredicting image {} ...".format(fn))
        key, slice_id = slices[i]
        normalizedImg = np.zeros(((1, n_size, n_size)))
        frame = np.zeros(((args.channels, n_size, n_size)))

        for j in range(args.channels):
            frame[j, :, :] = cv2.resize(in_files[key][slice_id-args.channels+j, :, :], (n_size, n_size), interpolation=cv2.INTER_CUBIC)
            if args.input == 'mask':
                frame[j,:,:] = cv2.GaussianBlur(frame[j,:,:], (7, 7), 0)
                frame[j,:,:] = cv2.normalize(frame[j,:,:], normalizedImg, 0, 255, cv2.NORM_MINMAX)
                ret, frame[j,:,:] = cv2.threshold(frame[j,:,:].astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
                frame[j,:,:] = cv2.normalize(frame[j,:,:], normalizedImg, 0, 1, cv2.NORM_MINMAX)
            if args.input == 'image':
                frame[j,:,:] = cv2.normalize(frame[j,:,:],  np.zeros(((1,n_size,n_size))), 0, 255, cv2.NORM_MINMAX)
        if args.net == 'ConvLSTM':
            frame = frame[..., np.newaxis]
            frame = torch.tensor(frame).permute(0, 3, 1, 2).contiguous()
        img = torch.Tensor(frame).clone().detach()
        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           device=device)
        out_files[i] = mask
        logging.info('Mask saved!')
    f = h5py.File(f'/content/drive/MyDrive/BIA2022/results/predictions_{args.name}.h5', mode='w')
    f.create_dataset('predictions',data = out_files)
    f.close()

