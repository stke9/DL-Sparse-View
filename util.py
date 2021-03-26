import torch
import numpy as np

def prep_img(img, phases=(0, 0.194, 0.233)):
    cutoffs = [np.mean([ph1, ph2]) for ph1, ph2 in zip(phases[1:], phases[:-1])]
    img[img < cutoffs[0]] = phases[0]
    img[img > cutoffs[-1]] = phases[-1]
    for lb, ub, ph in zip(cutoffs[:-1], cutoffs[1:], phases[1:-1]):
        img[np.logical_and(img < ub, img > lb)] = ph
    return img

def one_hot_y(img, phases=(0, 0.194, 0.233)):
    b, c, x, y = img.shape
    oh_img = torch.zeros([b, 4, x, y])
    for i, ph in enumerate(phases):
        oh_img[:, i][img[:, 0] == ph] = 1
    oh_img[:, -1][oh_img.mean(dim=1) == 0] = 1
    return oh_img

