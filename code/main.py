import os
from collections import defaultdict

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

from loss.triplet import TripletLoss, HTripletLoss, HierarchicalLoss
from models.hyper_resnet import HResNet
from models.resnet import ResNet
from models.rpn import ProposalNetwork
from utils.sample_utils import *
from utils.writer import Writer
from utils.data_lvis import DataSetWrapper

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        # a pretrained class agnostic maskrcnn model
        self.rpn = ProposalNetwork(self.device, self.config["rpn_weights"])
        dataset = DataSetWrapper(config["batch_size"], cfg=self.rpn.cfg)
        self.train_loader = dataset.get_train_loader()
        if self.config['loss']['include_hierarchical']:
            self.hierar_loss_crit = HierarchicalLoss(margin=config['loss']['margin']).cuda()
        if self.config['hyperbolic']:
            self.triplet_loss_crit = HTripletLoss(margin=config['loss']['margin']).cuda()
        else:
            self.triplet_loss_crit = TripletLoss(margin=config['loss']['margin']).cuda()
            
    def _get_device(self):
        device = self.config["device_id"]
        print("Running on:", device)
        return device

    def _init_model_and_optimizer(self):
        device_id = self._get_device()
        if self.config['hyperbolic']:
            model = nn.DataParallel(HResNet(**self.config["model"]), device_ids=[device_id])
        else:
            model = nn.DataParallel(ResNet(**self.config["model"]), device_ids=[device_id])
        model, loaded_iter = self._load_pre_trained_weights(model)

        lr = float(self.config['lr'])
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr,
            weight_decay=eval(self.config['weight_decay']))

        self.model = model
        self.optimizer = optimizer
        return loaded_iter

    def _step(self, a, p, n, loss_type):
        res = defaultdict(float)
        z_a = self.model(a.unsqueeze(0))[1]
        z_p = self.model(p.unsqueeze(0))[1]
        if n is not None:
            z_n = self.model(n.unsqueeze(0))[1]
        if loss_type == 'mask':
            res['mask_loss_count'] = 1
            res['mask_loss'] = self.triplet_loss_crit(z_a, z_p, z_n)
        elif loss_type == 'object':
            res['object_loss_count'] = 1
            res['object_loss'] = self.triplet_loss_crit(z_a, z_p, z_n)
        elif loss_type == 'hierar':
            res['hierar_loss_count'] = 1
            res['hierar_loss'] = self.hierar_loss_crit(z_a, z_p)
        else:
            raise Exception()
        return res
        
    def train(self):
        self.writer = Writer(self.config)
        train_loader = self.train_loader
        loaded_iter = self._init_model_and_optimizer()
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=60000, eta_min=0, last_epoch=-1)

        n_iter = loaded_iter + 1
        for epoch_counter in range(self.config['epochs']):
            for _, batch in enumerate(train_loader):
                image = [batch[i]['image'].to(self.device) for i in range(len(batch))]
                masks, boxes = [], []
                for i in range(len(image)):
                    masks_i, boxes_i = self.rpn(image[i], is_train=True)
                    masks.append(masks_i)
                    boxes.append(boxes_i.tensor.int())
                image_url = [batch[i]['file_name'] for i in range(len(batch))]
                assert (image[0].shape[2] == 3)

                loss_dict = {
                    'object_loss': 0, 'mask_loss': 0,  'hierar_loss': 0, 
                    'object_loss_count': 0, 'mask_loss_count': 0, 'hierar_loss_count': 0
                }
                mask_tensors = self._get_mask_tensors(image, masks, boxes)  # N, H, W, 3
                features = self._get_mask_features(mask_tensors)
  
                if self.config["loss"]["mask_loss"]:
                    triplets = sample_triplets_for_mask_loss_batched(masks, boxes, image, k=10)
                    for a, p, n in triplets:
                        res = self._step(a, p, n, loss_type='mask')
                        loss_dict = {k: v + res[k] for k, v in loss_dict.items()}

                if self.config["loss"]["object_loss"]:
                    triplets = sample_triplets_for_object_loss_batched(masks, boxes, image, k=2)
                    for a, p, n in triplets:
                        z_a = features[a[0]][a[1]]
                        z_p = features[p[0]][p[1]]
                        z_n = features[n[0]][n[1]]
                        loss_dict['object_loss_count'] += 1
                        loss_dict['object_loss'] += self.triplet_loss_crit(z_a, z_p, z_n)

                if self.config["loss"]["include_hierarchical"]:
                    pairs = sample_triplets_for_hierar_loss_batched(masks, boxes, image)
                    for p, c in pairs:
                        z_p = features[p[0]][p[1]]
                        z_c = features[c[0]][c[1]]
                        loss_dict['hierar_loss_count'] += 1
                        loss_dict['hierar_loss'] += self.hierar_loss_crit(z_p, z_c)

                total_loss = loss_dict['object_loss'] + self.config['hierar_loss_weight'] * loss_dict['hierar_loss'] + \
                             self.config['mask_loss_weight'] * loss_dict['mask_loss']
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if n_iter % self.config['log_loss_every_n_steps'] == 1:
                    print('Iter:', n_iter)
                    print(loss_dict)
                    self.writer.log_loss(loss_dict, n_iter)
                if n_iter % self.config['log_every_n_steps'] == 1 and masks[0].shape[0] > 1:
                    self.writer.visualize(image[0], image_url[0], masks[0], n_iter)
                if n_iter % self.config['save_checkpoint_every_n_steps'] == 0 and n_iter > 0:
                    print('Saving model..')
                    torch.save(self.model.state_dict(), os.path.join(self.writer.checkpoint_dir, 'model_'+str(n_iter)+'.pth'))

                n_iter +=1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()

            self.writer.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _get_mask_tensors(self, image, masks, boxes):
        mask_tensors = [
            torch.stack([
                crop(image[b], boxes[b][i], masks[b][i])
                for i in range(masks[b].shape[0])
            ])
            for b in range(len(image))
        ]
        return mask_tensors

    def _get_mask_features(self, tensors):
        features = []
        for b in range(len(tensors)):
            r, z = self.model(tensors[b])
            features.append(z)
        return features

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_files = os.listdir(self.writer.checkpoint_dir)
            saved_iters = [int(c.strip('.pth')[6:]) for c in checkpoints_files]
            loaded_iter = max(saved_iters) if len(saved_iters) > 0 else 0
            print('Found saved checkpoints at iter:', saved_iters)
            state_dict = torch.load(os.path.join(self.writer.checkpoint_dir, 'model_{}.pth'.format(loaded_iter)))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model at iteration {} with success.".format(loaded_iter))
        except FileNotFoundError:
            loaded_iter = 0
            print("Pre-trained weights not found. Training from scratch.")
        return model, loaded_iter

