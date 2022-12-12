#from re import T
#from select import kevent
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
from utils import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torchvision.transforms as T
from pycocotools.coco import COCO


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.lower()

        assert self.split in {'train', 'test'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        #with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
        with open(os.path.join(data_folder, 'images/', self.split), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, 'labels/', self.split), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


    
class COCODataset(Dataset):
    '''docstring'''
    # Adapted from PascalVOCDataset and this tutorial for COCO: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
    # and https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5

    def __init__(self, data_folder, split, keep_difficult=False):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            :param data_folder: folder where data files are stored
            :param split: split, one of 'TRAIN' or 'TEST'
            :param keep_difficult: keep or discard objects that are considered difficult to detect?
            json: coco annotation file path.
        """
        self.data_folder = data_folder
        self.coco = COCO()
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.keep_difficult = keep_difficult
        
        # Find data folder
        self.split = split.lower()

        assert self.split in {'train', 'val', 'test'}
        
        train_im_path = data_folder + '/train/im/'
        train_ann_path = data_folder + '/train/ann/'
        
        val_im_path = data_folder + '/val/im/'
        val_ann_path = data_folder + '/val/ann/'

        test_im_path = data_folder + '/test/im/'
        test_ann_path = data_folder + '/test/ann/'
        
        # self.filenames : create depending on split
        if self.split == 'train':
            self.images = [os.path.join(train_im_path,im_file) for im_file in os.listdir(train_im_path) if not im_file.startswith('.')]
            self.anns = [os.path.join(train_ann_path,obj_file) for obj_file in os.listdir(train_ann_path) if not obj_file.startswith('.')]
        elif self.split == 'val':
            self.images = [os.path.join(val_im_path,im_file) for im_file in os.listdir(val_im_path) if not im_file.startswith('.')]
            self.anns = [os.path.join(val_ann_path,obj_file) for obj_file in os.listdir(val_ann_path) if not obj_file.startswith('.')]
        else:
            self.images = [os.path.join(test_im_path,im_file) for im_file in os.listdir(test_im_path) if not im_file.startswith('.')]
            self.anns = [os.path.join(test_ann_path,obj_file) for obj_file in os.listdir(test_ann_path) if not obj_file.startswith('.')]

        self.images = sorted(self.images)
        self.anns = sorted(self.anns)
        
        assert len(self.images) == len(self.anns)
        

    def __getitem__(self, i):
        """Returns one data point (image, boxes, labels, difficulties)."""
        coco = self.coco
        
        
        image = self.images[i]
        ann = self.anns[i]
        
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        im_sz_y,im_sz_x = image.size
        
        with open(ann) as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            f.close()
        bb = np.zeros((len(lines),4))
        labels = []
        if len(lines)>0:
            for l_i, line in enumerate(lines):
                # Read line, save as pixel coordinates
                # Each line is category, minx, miny, maxx, maxy
                line = [float(num) for num in line]
                labels.append(1) # line[0] - but now artificially call it category 1, since we only have one
                bb_min_x = line[1]
                bb_min_y = line[2] 
                bb_max_x = line[3]
                bb_max_y = line[4] 
                bb[l_i,:] = [bb_min_x,bb_min_y,bb_max_x,bb_max_y]

            difficulties = np.zeros((len(labels),1))
        else: # make up object if image has none
            bb = [0,0,0.01,0.01]
            labels = [2] # artificial class
            difficulties = [0]
        
        boxes = torch.FloatTensor(bb)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)
        difficulties = torch.ByteTensor(difficulties)  # (n_objects)

        #tens_transf = T.ToTensor()
        #image = tens_transf(image)
        
        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

   
        return image, boxes, labels, difficulties
               
    def __len__(self):
        return len(self.images)        
            
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    
class BrainDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.lower()

        assert self.split in {'train', 'val', 'test'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        train_im_path = data_folder + 'images/train/'
        train_label_path = data_folder + 'labels/train/'
        
        val_im_path = data_folder + 'images/val/'
        val_label_path = data_folder + 'labels/val/'

        test_im_path = data_folder + 'images/test/'
        test_label_path = data_folder + 'labels/test/'

        # Get list om images and objects
        if self.split == 'train':
            self.images = [os.path.join(train_im_path,im_file) for im_file in os.listdir(train_im_path) if not im_file.startswith('.')]
            self.objects = [os.path.join(train_label_path,obj_file) for obj_file in os.listdir(train_label_path) if not obj_file.startswith('.')]
        elif self.split == 'val':
            self.images = [os.path.join(val_im_path,im_file) for im_file in os.listdir(val_im_path) if not im_file.startswith('.')]
            self.objects = [os.path.join(val_label_path,obj_file) for obj_file in os.listdir(val_label_path) if not obj_file.startswith('.')]
        else:
            self.images = [os.path.join(test_im_path,im_file) for im_file in os.listdir(test_im_path) if not im_file.startswith('.')]
            self.objects = [os.path.join(test_label_path,obj_file) for obj_file in os.listdir(test_label_path) if not obj_file.startswith('.')]

        self.images = sorted(self.images)
        self.objects = sorted(self.objects)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        im_sz_y,im_sz_x = image.size 

        # Read objects in this image (bounding boxes, labels, difficulties)
        # objects = self.objects[i]
        # boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        # labels = torch.LongTensor(objects['labels'])  # (n_objects)
        # #difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)
        
        labels = [];
        with open(self.objects[i]) as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            f.close()
        bb = np.zeros((len(lines),4))
        for l_i, line in enumerate(lines):
            # Read line, save as pixel coordinates
            line = [float(num) for num in line]
            bb_sz_x = line[4]*im_sz_x
            bb_sz_y = line[3]*im_sz_y
            bb_min_x = line[2]*im_sz_x
            bb_min_y = line[1]*im_sz_y
            bb_max_x = line[2]*im_sz_x + bb_sz_x
            bb_max_y = line[1]*im_sz_y + bb_sz_y
            bb[l_i,:] = [bb_min_x,bb_min_y,bb_max_x,bb_max_y]
            labels.append(1)

        difficulties = np.zeros((len(labels),1))
        difficulties = torch.ByteTensor(difficulties)  # (n_objects)
        
        image, bb = self.pad2square(image, bb)
        
        
        # Resize bounding boxes
        # target_im_size = (300,300)
        # im_orig_sz = image.size[0]
        # resize_factor = target_im_size[0]/im_orig_sz
        # bb = bb*resize_factor

        

        

        boxes = torch.FloatTensor(bb)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)


        # Plot image and bounding box
        # ax=plt.subplot(1,1,1)
        # plt.axis('off')
        # plt.imshow(image, cmap='gray')
        # for b_i in range(0,l_i+1):
        #     # Create a Rectangle patch
        #     bb_sz_x = bb[b_i,2]-bb[b_i,0]
        #     bb_sz_y = bb[b_i,3]-bb[b_i,1]
        #     bb_pos_x = bb[b_i,0] - bb_sz_x/2 # center of box
        #     bb_pos_y = bb[b_i,1] - bb_sz_y/2 # center of box
        #     bb_plot = patches.Rectangle((bb_pos_y, bb_pos_x), bb_sz_y, bb_sz_x, linewidth=1, edgecolor='r', facecolor='None')
        #     #bb_plot = patches.Rectangle((10, 10), 10, 10, linewidth=1, edgecolor='r', facecolor='None')

        #     # Add the patch to the Axes
        #     ax.add_patch(bb_plot)
        # plt.show()

        # Apply transformations
        #image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        image, boxes = resize(image, boxes, dims=(300, 300))
        
        

        tens_transf = T.ToTensor()
        image = tens_transf(image)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        image = T.functional.normalize(image, mean=mean, std=std)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


    def pad2square(self,image,bb):
        max_size = max(image.size)
        pad_left,pad_top = [(max_size - s)//2 for s in image.size]
        pad_right,pad_bottom = [(max_size - (s + pad)) for s, pad in zip(image.size, [pad_left, pad_top])]
        pad = [pad_left, pad_top, pad_right, pad_bottom]
        bb[:,(1,3)] += pad_left
        bb[:,(0,2)] += pad_bottom
        image = T.functional.pad(image, pad, 0, 'constant')
        return image,bb


    










