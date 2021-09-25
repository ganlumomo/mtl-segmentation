"""
Null Loader
"""


"""
RELLIS-3D Off Road Dataset Loader
"""

import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data
import logging
import datasets.uniform as uniform
import datasets.northcampusforrest_labels as northcampusforrest_labels
import json
from config import cfg

import random


trainid_to_name = northcampusforrest_labels.trainId2name
id_to_trainid = northcampusforrest_labels.label2trainid
trainid_to_color = northcampusforrest_labels.trainId2color
num_classes = 19
ignore_label = 0
root = cfg.DATASET.NORTHCAMPUSFORREST_DIR

palette = []
for i in range(num_classes):
    palette.append(trainid_to_color[i][0])
    palette.append(trainid_to_color[i][1])
    palette.append(trainid_to_color[i][2])
#print(palette)

zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def get_item_and_mask(file_path):

    # get sorted list of image file names
    ims = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(file_path,"images"))])
    mks = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(file_path,"labels_id"))])
    
    imsWithLabels = list(set(ims) & set(mks))

    ims = [os.path.join(file_path, "images", i + ".png") for i in imsWithLabels]
    mks = [os.path.join(file_path, "labels_id", i + ".png") for i in imsWithLabels]

    data_set = []
    
    q = 0
    for item in ims: 
        data_set.append((item, mks[q])) 
        #print(data_set[q])
        q+=1

    return data_set

def make_dataset(quality, mode, maxSkip=0, cv_split=0, hardnm=0):
    items = []
    all_items = []
    aug_items = []

    assert mode in ['train', 'val', 'test', 'trainval']

    # load train/val/test data

    train_path = os.path.join(root, 'training')
    train_set = get_item_and_mask(train_path)
    val_path = os.path.join(root, 'testing')
    val_set = get_item_and_mask(val_path)

    if mode == 'train':
        items = train_set
    elif mode == 'val':
        items = val_set
    elif mode == 'test':
        items = test_set
    elif mode == 'trainval':
        items = train_set + val_set
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('North Cammpus Forrest-{}: {} images'.format(mode, len(items)))

    return items, aug_items

def make_test_dataset(quality, mode, maxSkip=0, cv_split=0):
    items = []
    # assert quality == 'semantic'
    assert mode == 'test'

    test_path = os.path.join(root, 'test.lst')
    item = get_item_and_mask(test_path)
    logging.info('North Campus Forrest has a total of {} test images'.format(len(items)))

    return items, []

class NorthCampusForrest(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=0, test=False,
                 cv_split=None, scf=None, hardnm=0):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.scf = scf
        self.hardnm = hardnm

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS)
        else:
            self.cv_split = 0

        if self.mode == 'test':
            self.imgs, _ = make_test_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split)
        else:
            self.imgs, _ = make_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split, hardnm=self.hardnm)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:
            if self.scf:
                json_fn = 'ncf_tile{}_cv{}_scf.json'.format(self.class_uniform_tile, self.cv_split)
            else:
                json_fn = 'ncf_tile{}_cv{}_{}_hardnm{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode, self.hardnm)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                if self.scf:
                    self.centroids = kitti_uniform.class_centroids_all(
                        self.imgs,
                        num_classes,
                        id2trainid=id_to_trainid,
                        tile_size=class_uniform_tile)
                else:
                    self.centroids = uniform.class_centroids_all(
                        self.imgs,
                        num_classes,
                        id2trainid=id_to_trainid,
                        tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)

        self.build_epoch()

    def build_epoch(self, cut=False):
        if self.class_uniform_pct > 0:
            self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                    self.centroids,
                                                    num_classes,
                                                    cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem

        if self.mode == 'test':
            img, mask = Image.open(img_path).convert('RGB'), None
        else:
            img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]


        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # resize images
        width, height = 640, 480
        img = img.resize((width, height), Image.BICUBIC)
        mask = mask.resize((width, height), Image.NEAREST)

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:
            outdir = './dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)
            if self.mode == 'test':
                img_keepsize = self.transform(img_keepsize)
                mask = img_keepsize
        if self.target_transform is not None:
            if self.mode != 'test':
                mask = self.target_transform(mask)

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)





'''
import numpy as np
import torch
from torch.utils import data
from config import cfg
from PIL import Image
import os

num_classes = 21
ignore_label = 255
root = cfg.DATASET.NORTHCAMPUSFORREST_DIR

class NorthCampusForrest(data.Dataset):
    """
    North Campus Dataset
    """
    def __init__(self, mode, joint_transform_list=None,
                 transform=None, target_transform=None):
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform

        print('debug')
        if(mode == "train"):
            self.imgFolder = os.path.join(root, "training")
        elif(mode == "val"):
            self.imgFolder = os.path.join(root, "testing")
        else:
            raise RuntimeError("Mode must be train or val")

        # get sorted list of image file names
        self.images = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.imgFolder,"images"))])
        self.masks = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.imgFolder,"labels_id"))])
        
    def __getitem__(self, index):
    	#Return img, mask, name
        img = Image.open( os.path.join(self.imgFolder, "images", self.images[index] + ".png") ).convert('RGB')
        mask = Image.open( os.path.join(self.imgFolder, "labels_id", self.masks[index] + ".png") ).convert('RGB')
        
        # resize images
        width, height = 640, 480
        img = img.resize((width, height), Image.BICUBIC)
        mask = mask.resize((width, height), Image.NEAREST)

        # Image Transformations
        centroid = None
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)
        
        # convert to np
        #img = np.asarray(img)
        #mask = np.asarray(mask)
        #print("IMG Size")
        #print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
            #if self.mode == 'test':
            #    img_keepsize = self.transform(img_keepsize)
            #    mask = img_keepsize
        if self.target_transform is not None:
            #if self.mode != 'test':
            mask = self.target_transform(mask)

        return img, mask, self.images[index]


    def __len__(self):
        return len(self.images)


'''