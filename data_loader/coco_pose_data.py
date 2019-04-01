"""
This is the data loader of Hand Pose datasets to train or test CPM model

There are four variables in one item
image   Type: Tensor    Size: 3 * 368 * 368
label   Type: Tensor    Size: 21 * 45 * 45
center  Type: Tensor    Size: 3 * 368 * 368
name    Type:  str

The data is organized in the following style

----data                        This is the folder name like train or test
--------431077.jpg
-------- .....

----label                        This is the folder name like train or test
--------data_label.json          This is one sequence of images

To have a better understanding, you can view ../dataset in this repo
"""
import os
import sys
import json
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np

import scipy.misc

sys.path.append('../')
from src.util import *

class COCOPoseDataset(Dataset):
    def __init__(self, data_dir, label_dir, joints=18, transform=None, sigma=1):
        self.height = 368
        self.width = 368

        self.data_dir = data_dir
        self.label_dir = label_dir

        self.transform = transform
        self.joints = joints  # 18 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        self.images_dir = []
        self.gen_imgs_dir()

    def gen_imgs_dir(self):
        """
        get directory of all images
        :return:
        """

        imgs = os.listdir(self.data_dir)  # [431077.jpg, ......]
        for img in imgs:               
            if img == '.DS_Store':
                continue
            self.images_dir.append(self.data_dir + '/' + img)  #

        print 'total numbers of image is ' + str(len(self.images_dir))

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        images          3D Tensor      3                *   height(368)      *   weight(368)
        label_map       3D Tensor      (joints + 1)     *   label_size(45)   *   label_size(45)
        """

        label_size = self.width / 8 - 1         # 45
        img = self.images_dir[idx]              # '.../001L0/L0005.jpg'

        labels = json.load(open(self.label_dir + '/data_label.json'))

        # get image
        im = Image.open(img)                # read image
        w, h, c = np.asarray(im).shape      # weight 256 * height 256 * 3
        ratio_x = self.width / float(w)
        ratio_y = self.height / float(h)    # 368 / 256 = 1.4375
        im = im.resize((self.width, self.height))                       # unit8      weight 368 * height 368 * 3
        image = transforms.ToTensor()(im)   # 3D Tensor  3 * height 368 * weight 368

        # get label map
        label = labels[img.split('/')[-1][:-4]]
        lbl = self.genLabelMap(label, label_size=label_size, joints=self.joints, ratio_x=ratio_x, ratio_y=ratio_y)
        label_maps = torch.from_numpy(lbl)

        return image.float(), label_maps.float(), img

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def genLabelMap(self, label, label_size, joints, ratio_x, ratio_y):
        """
        generate label heat map
        :param label:               list            21 * 2
        :param label_size:          int             45
        :param joints:              int             21
        :param ratio_x:             float           1.4375
        :param ratio_y:             float           1.4375
        :return:  heatmap           numpy           joints * boxsize/stride * boxsize/stride
        """
        # initialize
        label_maps = np.zeros((joints, label_size, label_size))
        background = np.zeros((label_size, label_size))

        # each joint
        for i in range(len(label)):
            lbl = label[i]                      # [x, y]
            x = lbl[0] * ratio_x / 8.0          # modify the label
            y = lbl[1] * ratio_y / 8.0
            heatmap = self.genCenterMap(y, x, sigma=self.sigma, size_w=label_size, size_h=label_size)  # numpy
            background += heatmap               # numpy
            label_maps[i, :, :] = np.transpose(heatmap)  # !!!

        return label_maps  # numpy           label_size * label_size * (joints + 1)


# test case
if __name__ == "__main__":
    data_dir = '/Users/huangju/Desktop/train_data/data'
    label_dir = '/Users/huangju/Desktop/train_data/label'
    data = COCOPoseDataset(data_dir=data_dir, label_dir=label_dir)

    img, label, name = data[1]
    print 'dataset info ... '
    print img.shape         # 3D Tensor 3 * 368 * 368
    print label.shape       # 3D Tensor 21 * 45 * 45
    print name              # str   ../dataset/train_data/001L0/L0461.jpg

    # ***************** draw label map *****************
    print 'draw label map ....'
    lab = np.asarray(label)
    out_labels = np.zeros(((45, 45)))
    for i in range(18):
        out_labels += lab[i, :, :]
    scipy.misc.imsave('img/coco_label.jpg', out_labels)

    # ***************** draw image *****************
    print 'draw heat map ....'
    im_size = 368
    img = transforms.ToPILImage()(img)
    img.save('img/coco_img.jpg')
    heatmap = np.asarray(label[0, :, :])

    im = Image.open('img/coco_img.jpg')

    heatmap_image(img, lab, joint_num=18, save_dir='img/coco_heat.jpg')








