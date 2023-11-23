# coding=utf-8
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import pydicom


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        dcm_info = pydicom.read_file(self.dataset_path+'/your_data_path/'+name + '.dcm',force=True)
        dcm = dcm_info.pixel_array # 提取图像信息

        png = cv2.imread(self.dataset_path+'/your_mask_path/'+name + '.png', cv2.COLOR_BGR2GRAY)

        dcm,png = self.self_normalization(dcm,png)

        dcm = np.tile(dcm,(3,1)).reshape((3,800,800))

        seg_labels = np.eye(self.num_classes)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes))

        return dcm, png, seg_labels

    def self_normalization(self,dcm,png):

        png[png == 255] = 1

        max_dcm = np.max(dcm)
        min_dcm = np.min(dcm)
        gap = max_dcm - min_dcm

        if gap != 0:
            dcm = (dcm - min_dcm) / gap
        else:
            dcm[:] = 0
            dcm = dcm/1.0

        return dcm,png

def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
