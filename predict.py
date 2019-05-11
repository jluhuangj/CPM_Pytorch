"""
Predict 21 key points for images without ground truth
step 1: predict label and save into json file for every image
"""
import os
import sys
import json
import ConfigParser
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader.coco_pose_data import COCOPoseDataset
from model.cpm import CPM


# *********************** hyper parameter  ***********************

config = ConfigParser.ConfigParser()
config.read('conf.text')

batch_size = config.getint('predict', 'batch_size')
best_model = config.getint('test', 'best_model')

predict_data_dir    = config.get('predict', 'predict_data_dir')
save_label_dir      = config.get('predict', 'save_label_dir')
save_heatmap_dir    = config.get('predict', 'save_heatmap_dir')

if not os.path.exists(save_label_dir):
    os.mkdir(save_label_dir)

cuda = torch.cuda.is_available()

# *********************** function ***********************

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def heatmap_image(img, label, save_dir = save_label_dir):
    """
    draw heat map of each joint
    :param img:             a PIL Image
    :param heatmap          type: numpy     size: 21 * 45 * 45
    :return:
    """
    im_size = 64

    img = img.resize((im_size, im_size))
    x1 = 0
    x2 = im_size

    y1 = 0
    y2 = im_size

    target = Image.new('RGB', (6 * im_size, 3 * im_size))
    for i in range(18):
        heatmap = label[i, :, :]    # heat map for single one joint

        # remove white margin
        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        fig = plt.gcf()

        fig.set_size_inches(7.0 / 3, 7.0 / 3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(heatmap)
        plt.text(10, 10, '{0}'.format(i), color='r', fontsize=24)
        plt.savefig('tmp.jpg')

        heatmap = Image.open('tmp.jpg')
        heatmap = heatmap.resize((im_size, im_size))

        img_cmb = Image.blend(img, heatmap, 0.5)

        target.paste(img_cmb, (x1, y1, x2, y2))

        x1 += im_size
        x2 += im_size

        if i == 5 or i == 11:
            x1 = 0
            x2 = im_size
            y1 += im_size
            y2 += im_size

    target.save(save_dir)
    os.system('rm tmp.jpg')


def save_label(predict_heatmaps, imgs, image=''):
    """
    :param predict_heatmaps:    4D Tensor    batch size * 21 * 45 * 45
    :param imgs:                batch_size * 1
    :return:
    """
    for b in range(predict_heatmaps.shape[0]):  # for each batch (person)
        img_name = imgs[b].split('/')[-1]

        labels_list = []  # 18 points label for one image [[], [], [], .. ,[]]

        # ****************** save image and label of 21 joints ******************
        for i in range(18):  # for each joint
            tmp_pre = np.asarray(predict_heatmaps[b, i, :, :].data)  # 2D
            #  get label of original image
            corr = np.where(tmp_pre == np.max(tmp_pre))
            x = corr[0][0] * (256.0 / 45.0)
            x = int(x)
            y = corr[1][0] * (256.0 / 45.0)
            y = int(y)
            labels_list.append([y, x])  # save img label to json

        #print labels_list
        labels_str = ','.join([','.join([str(x) for x in point]) for point in labels_list])
        #print labels_str

        # ****************** save label ******************
        save_dir_label = os.path.join(save_label_dir, img_name + '.txt')
        with open(save_dir_label, 'w') as f:
            f.write(labels_str)

        if save_heatmap_dir != '':
            import cv2
            image = image.numpy()
            cv2.imwrite(os.path.join(save_heatmap_dir, img_name), image)
            for point in labels_list:
                image = cv2.circle(image, tuple(point), 6, (0, 0, 255), thickness=-1)
            #cv2.imwrite(os.path.join(save_heatmap_dir, img_name), image)


# ************************************ Build dataset ************************************

test_data = COCOPoseDataset(data_dir=predict_data_dir)
print 'Test dataset total number of images is ----' + str(len(test_data))

# Data Loader
test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Build model
net = CPM(18)
if cuda:
    net = net.cuda()

model_path = os.path.join('ckpt/model_epoch' + str(best_model)+'.pth')
state_dict = torch.load(model_path)
net.load_state_dict(state_dict)

# **************************************** test all images ****************************************
print '********* test data *********'
net.eval()

for step, (image, imgs) in enumerate(test_dataset):
    image_cpu = image
    image = Variable(image.cuda() if cuda else image)   # 4D Tensor
    # Batch_size  *  3  *  width(368)  *  height(368)

    print imgs
    pred_6 = net(image)  # 5D tensor:  batch size * stages(6) * 41 * 45 * 45

    # ****************** from heatmap to label ******************
    save_label(pred_6[:, 5, :, :, :], imgs=imgs, image = image_cpu)
    #sys.exit()

    # ****************** draw heat maps ******************
    if save_heatmap_dir != '':
        for b in range(image_cpu.shape[0]):
            img = image_cpu[b, :, :, :]         # 3D Tensor
            img = transforms.ToPILImage()(img.data)        # PIL Image
            pred = np.asarray(pred_6[b, 5, :, :, :].data)      # 3D Numpy
            #pred = pred_6[b, 5, :, :, :].cpu().detach().numpy()      # 3D Numpy

            img_name = imgs[b].split('/')[-1]
            if not os.path.exists(save_heatmap_dir):
                os.mkdir(save_heatmap_dir)
            img_dir = save_heatmap_dir + '/' + img_name + '.jpg'
            heatmap_image(img, pred, save_dir=img_dir)

print 'success...'

