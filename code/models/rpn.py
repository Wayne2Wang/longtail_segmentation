import torch
import torch.nn as nn
import numpy as np
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer

# Config file.
CFG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
OUTPUT_DIR = "./"

def get_modelzoo_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
    return cfg


def get_class_agnostic_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
    # Explicitly makes the model to have class-agnostic mask branch.
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    # Load the default pretrained weights.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
    return cfg


class ProposalNetwork(nn.Module):
    def __init__(self, device, checkpoint_path=None, nms_thres=0.6, post_nms_topk=20, pre_nms_topk=30):
        super(ProposalNetwork, self).__init__()
        self.cfg = get_class_agnostic_config()
        
        # some parameters to play with
        self.cfg.MODEL.RPN.NMS_THRESH = nms_thres
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = pre_nms_topk
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = post_nms_topk
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = pre_nms_topk
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = post_nms_topk
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.cfg.SOLVER.IMS_PER_BATCH = 6
        if checkpoint_path is not None:
            self.cfg.MODEL.WEIGHTS = checkpoint_path
            print("Overwrite rpn weights with weights in", checkpoint_path)
        
        self.predictor = DefaultPredictor(self.cfg)
        print('Build Predictor using cfg')

    def train_predictor(self):
        """ Train a region proposal network using Detectron2's trainer.
        """
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 5000
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR /= 2.
        self.predictor = DefaultPredictor(self.cfg)
        self.cfg.OUTPUT_DIR = OUTPUT_DIR
        print(self.cfg.OUTPUT_DIR)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def save(self, path, i):
        checkpoints_folder = os.path.join(path, 'rpn_model_'+str(i)+'.pth')
        torch.save(self.predictor.model.state_dict(), checkpoints_folder)

    def load(self, checkpoints_folder, i):
        try:
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_'+str(i)+'.pth'))
            self.predictor.model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    def forward(self, x, is_train=False):
        """ Takes the raw image, and then outputs the boxes and the class agnostic masks
        :param x: (h, w, 3) tensor
        :return: (topk, h, w), (h, w)
        """
        x = x.cpu().numpy()
        x = x.astype(np.uint8)  # [0, 255]
        assert(x.shape[2] == 3)
        out = self.predictor(x)  # predictor takes images in the BGR format
        if is_train:
            masks = out['instances'].pred_masks
            boxes = out['instances'].pred_boxes
            return masks, boxes
        else:
            # do nms using the mask iou.
            return [out]
        

if __name__ == '__main__':
    rpn = ProposalNetwork('cuda')
    rpn.train_predictor()
