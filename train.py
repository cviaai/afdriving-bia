#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from unet import UNet, ConvLSTM
from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_default = 'data/'
dir_checkpoint = '/content/drive/MyDrive/BIA2022/checkpoints/'

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

def train_net(net,
              device,
              name,
              channels,
              tag = 'UNet',
              mode = 'sampled',
              input='image',
              epochs=150,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):
    logging.info(f'Using mode {mode}')
    dataset = BasicDataset(channels,mode,input,tag)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    writer = SummaryWriter(f'/content/drive/MyDrive/BIA2022/logs{name}',
                           comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Name:            {name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mode:            {mode}
        Input:           {input}
        Channels:        {channels}
        Net:             {tag}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.out_channels > 1 else 'max', patience=2)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                assert imgs.shape[1] == net.in_channels,  f'Network has been defined with {net.in_channels} input channels, '                     f'but loaded images have {imgs.shape[1]} channels. Please check that '                     'the images are loaded correctly.'
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.out_channels == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                if tag == 'ConvLSTM':
                    masks_pred = net(imgs)
                else:
                    masks_pred = net(imgs)
          
                loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'{name}_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-c', '--channels', metavar='C', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, default='sampled',
                        help='Sampled - each 25 ms, Full - each slice')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        help='image, mask')
    parser.add_argument('-n','--name', dest='name', type=str, default='experiment_name',
                        help='name of the experiment')
    parser.add_argument('--model', dest='net', type=str, default='Unet',
                        help='Unet or ConvLSTM')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.net == 'Unet':
        net = UNet(in_channels=args.channels, out_channels=1, bilinear=True)
    if args.net == 'ConvLSTM':
        config = Config()
        net = ConvLSTM(config, in_channels=args.channels, out_channels=1).to(config.device)

    logging.info(f'Using mode {args.mode}')

    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n')
            

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  channels = args.channels,
                  epochs=args.epochs,
                  name=args.name,
                  tag = args.net,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  mode = args.mode,
                  input = args.input)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

