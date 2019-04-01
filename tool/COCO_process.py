# coding=utf-8
import os
import time
import ujson

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

'''
train2014  : 82783 simages
val2014    : 40504 images

first 2644 of val2014 marked by 'isValidation = 1', as our minval dataset.
So all training data have 82783+40504-2644 = 120643 samples
'''

class Cocokeypoints(Dataset):
    def __init__(self, root, data):
        # add preprocessing as a choice, so we don't modify it manually.
        self.data = data
        self.index_list = os.listdir(root)
        self.numSample = len(self.index_list)
        self.root = root

    def get_anno(self, meta_data):
        """
        get meta information
        """
        anno = dict()
        anno['dataset'] = meta_data['dataset']
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['people_index'] = int(meta_data['people_index'])
        anno['annolist_index'] = int(meta_data['annolist_index'])

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']
        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])
        anno['num_keypoints_other'] = np.array(
            meta_data['num_keypoints_other'])
        anno['joint_others'] = np.array(meta_data['joint_others'])
        anno['objpos_other'] = np.array(meta_data['objpos_other'])
        anno['scale_provided_other'] = meta_data['scale_provided_other']
        anno['bbox_other'] = meta_data['bbox_other']
        anno['segment_area_other'] = meta_data['segment_area_other']

        if anno['numOtherPeople'] == 1:
            anno['joint_others'] = np.expand_dims(anno['joint_others'], 0)
            anno['objpos_other'] = np.expand_dims(anno['objpos_other'], 0)
        return anno

    def add_neck(self, meta):
        '''
        MS COCO annotation order:
        0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
        5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
        9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
        14: r knee		15: l ankle		16: r ankle

        The order in this work:
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                     12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # Index 6 is right shoulder and Index 5 is left shoulder
        right_shoulder = meta['joint_self'][6, :]
        left_shoulder = meta['joint_self'][5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 or left_shoulder[2] == 2:
            neck[2] = 2
        elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
            neck[2] = 1
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        meta['joint_self'] = np.vstack((meta['joint_self'], neck))
        meta['joint_self'] = meta['joint_self'][our_order, :]
        temp = []

        for i in range(meta['numOtherPeople']):
            right_shoulder = meta['joint_others'][i, 6, :]
            left_shoulder = meta['joint_others'][i, 5, :]
            neck = (right_shoulder + left_shoulder) / 2
            if (right_shoulder[2] == 2 or left_shoulder[2] == 2):
                neck[2] = 2
            elif (right_shoulder[2] == 1 or left_shoulder[2] == 1):
                neck[2] = 1
            else:
                neck[2] = right_shoulder[2] * left_shoulder[2]
            neck = neck.reshape(1, len(neck))
            neck = np.round(neck)
            single_p = np.vstack((meta['joint_others'][i], neck))
            single_p = single_p[our_order, :]
            temp.append(single_p)
        meta['joint_others'] = np.array(temp)

        return meta

    def resize(self, img, points, dst_x= 256, dst_y = 256):
        src_y = len(img)
        src_x = len(img[0])

        x_ratio = float(src_x)/dst_x
        y_ratio = float(src_y)/dst_y

        img = cv2.resize(img, (dst_x, dst_y))

        dst_points = []
        for point in points:
            x = int(point[0]/x_ratio)
            y = int(point[1]/y_ratio)
            dst_points.append([x, y])

        return img, dst_points
        
    def __getitem__(self, index):
        img_name = self.index_list[index]

        img_path = ''
        anno_data = dict()
        for one_data in self.data:
            if img_name in one_data['img_paths']:
                img_path = one_data['img_paths']
                anno_data = one_data
                break
        if img_path != '':
            img = cv2.imread(os.path.join(self.root, img_path.split('/')[-1]))

        meta_data = self.get_anno(anno_data)
        meta_data = self.add_neck(meta_data)
        img, points = self.resize(img, meta_data['joint_self'])

        #return img, meta_data
        #return img, meta_data['joint_self']
        return img_name, img, points

    def __len__(self):
        return self.numSample

if __name__ == '__main__':
    root_dir = '/Users/huangju/Desktop/train2014_one_person'
    json_path = '/Users/huangju/Desktop/COCO.json'
    w_path = '/Users/huangju/Desktop/train_data'
    if not os.path.exists(w_path + '/data'):
        os.makedirs(w_path + '/data')

    if not os.path.exists(w_path + '/label'):
        os.makedirs(w_path + '/label')

    start_time = time.time()
    with open(json_path) as f:
        f_data = f.read()
    read_time = time.time()

    json_data = ujson.loads(f_data)
    json_data = json_data['root']
    json_time = time.time()
    
    coco_data = Cocokeypoints(root_dir, json_data)
    dataset_time = time.time()

    w_dict = dict()
    for img_name, img, points in coco_data:
        w_name = img_name[-10:]

        img_path = w_path + '/data/' + w_name
        cv2.imwrite(img_path, img)

        w_dict[w_name[:-4]] = points
        print w_name[:-4]
    proc_time = time.time()

    w_data = ujson.dumps(w_dict)
    with open(w_path + '/label/data_label.json', 'w') as f:
        f.write(w_data)
    w_time = time.time()
       
    print 'read time: {0:.2f}s, \
           json time: {1:.2f}s, \
           dataset time: {2:.2f}s, \
           process time: {3:.2f}s, \
           write time: {4:.2f}s'.format(read_time - start_time,
                                       json_time - read_time,
                                       dataset_time - json_time,
                                       proc_time - dataset_time,
                                       w_time - proc_time)

    




