from functools import partial
from itertools import product 

import numpy as np
import torch
import torchvision.transforms as transforms

DIM = 224  # input size to the feature extractor
        

#####################################################
#### Batched sampling function called by trainer ####
#####################################################
def sample_triplets_for_mask_loss_batched(masks_batch, boxes_batch, images_batch, k):
    """
    masks (list[tensor]):
    boxes (list[numpy]):
    image (list[tensor]):
    k (int): number of triplets to sample for each image in the batch
    """
    batch_size = len(masks_batch)
    for b in range(batch_size):
        masks, boxes, image = masks_batch[b], boxes_batch[b], images_batch[b]
        n = masks.shape[0]
        idx = np.random.choice(range(n), k, replace=False)
        for i in idx:
            m, box = masks[i], boxes[i]
            full, fg, bg = crop(image, box), crop(image, box, m), crop(image, box, ~m)
            yield full, fg, bg


def sample_triplets_for_hierar_loss_batched(masks_batch, boxes_batch, images_batch):
    batch_size = len(masks_batch)
    for b in range(batch_size):
        masks, boxes, image = masks_batch[b], boxes_batch[b], images_batch[b]
        n = masks.shape[0]
        for i in range(n):
            m, box = masks[i], boxes[i]
#             parent = crop(image, box, m)
            child_flag = list(map(partial(is_child, m), masks))
            for j in np.where(child_flag)[0]:
                if (i == j): continue
#                 child = crop(image, boxes[j], masks[j])
                yield [b, i], [b, j]
        

def sample_triplets_for_object_loss_batched(masks_batch, boxes_batch, image_batch, k):
    batch_size = len(masks_batch)
    for b in range(batch_size):
        masks, boxes, image = masks_batch[b], boxes_batch[b], image_batch[b]
        n = masks.shape[0]
        for i in range(n):
            m, box = masks[i], boxes[i]
#             anchor = crop(image, box, m)

            pos_flags = list(map(partial(same_object, m), masks))
            pos_idx = np.where(pos_flags)[0]
        
            for j in pos_idx:
                if (i == j): continue
#                 positive = crop(image, boxes[j], masks[j])
                
                # negative sample can be from current image or other images in the batch
                neg_b_n = np.random.permutation(list(product(range(batch_size), range(n))))
                sample_count = 0
                for b_n in neg_b_n:
                    if b_n[0] == b and (b_n[1] == i or b_n[1] == j):
                        continue
                    if b_n[1] >= len(masks_batch[b_n[0]]):
                        continue
#                     negative = crop(image_batch[b_n[0]], boxes_batch[b_n[0]][b_n[1]], masks_batch[b_n[0]][b_n[1]])
                    
                    sample_count += 1
                    if sample_count == k: break
#                     yield anchor, positive, negative
                    yield [b,i], [b,j], b_n
                

                
##########################
#### helper functions ####
##########################

def mask_area(m):
    return m.sum().item()


def mask_iou(m1, m2):
    union = mask_area(m1 | m2)
    if not union > 0: return 0.
    res = mask_area(m1*m2) * 1. / union
    return res


def is_child(anchor, m):
    anchor_area, m_area = mask_area(anchor), mask_area(m)
    if mask_area(anchor*m) == m_area and anchor_area > m_area:
        return True
    return False


def same_object(anchor, m):
    return mask_iou(anchor, m) > 0.4


def crop(image, box, mask=None, square=True):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image/255.)
    if isinstance(image, torch.Tensor):
        if image.shape[2] == 3:
            image = image.permute(2,0,1)
        if mask is not None:
            image = image * mask.unsqueeze(0)
        image = transforms.ToPILImage()(image.cpu()/255.)
    else:
        raise Exception()
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()

    if square:
        h = box[2] - box[0]
        w = box[3] - box[1]
        center = ((box[3]+box[1])/2., (box[0]+box[2])/2.)
        new_len = max(h, w) / 2.
        box = [center[1]-new_len, center[0]-new_len, center[1]+new_len, center[0]+new_len]
#         print(box)
        
    cropped = image.crop((box[0], box[1], box[2], box[3]))
    resized = transforms.Resize((DIM, DIM))(cropped)
    return transforms.ToTensor()(resized).cuda()  # shape (3, dim, dim)