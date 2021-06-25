import json
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Writer(object):
    def __init__(self, config):
        self.config = config
        self.name = '{}_hyp={}_zdim={}_hierar={}_maskloss={}'.format(
            config['desc'], config['hyperbolic'], config['model']['out_dim'],
            config['loss']['include_hierarchical'], config['loss']['mask_loss']
        )
        checkpoint_dir = './checkpoints/{}'.format(self.name)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter('runs/{}'.format(self.name))
        config_str = json.dumps(config)
        self.writer.add_text('configs', config_str, 0)

    def add_scalar(self, text, val, i):
        self.writer.add_scalar(text, val, i)

    def log_loss(self, loss_dict, n_iter):
        total_loss = loss_dict['object_loss'] + self.config['hierar_loss_weight'] * loss_dict['hierar_loss'] + \
                     self.config['mask_loss_weight'] * loss_dict['mask_loss']

        for type in loss_dict.keys():
            self.writer.add_scalar(type, loss_dict[type], global_step=n_iter)
        self.writer.add_scalar('total_loss', total_loss, global_step=n_iter)
        for loss_type in ['object', 'mask', 'hierar']:
            if loss_dict[loss_type+'_loss_count'] > 0:
                self.writer.add_scalar(loss_type+'_loss_mean', 
                                       loss_dict[loss_type+'_loss']/loss_dict[loss_type+'_loss_count'], 
                                       global_step=n_iter)

    def visualize(self, image, image_url, masks, n_iter):
        h, w = masks[0].shape[0], masks[0].shape[1]
        proposed_cls = bin_to_cls_mask(masks.cpu().numpy(), plot=True)
        img = (image*255).type(torch.int32).cpu().numpy().transpose(2, 0, 1)  # [::-1,:,:]
        self.writer.add_image('input_images', img, n_iter)
        self.writer.add_text('filename', image_url, n_iter)
        self.writer.add_image('proposed_masks', make_grid(torch.tensor(proposed_cls.reshape(1, 1, h, w))), n_iter)

        
def bin_to_cls_mask(labels, plot=True):
    labels = labels.astype(bool)
    h, w = labels.shape[1:]
    mask = np.zeros((h, w))
    for i in reversed(range(labels.shape[0])):
        mask[labels[i]] = i+1
    if plot:
        mask = mask / (labels.shape[0]+1)*255  # convert to greyscale
    return mask.astype(np.uint8)
