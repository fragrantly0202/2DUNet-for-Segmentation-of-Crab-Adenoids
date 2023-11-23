# coding=utf-8
import os
import numpy as np
import torch
import cv2
from nets.unet import Unet
import pydicom
import torch.nn.functional as F

model_path = r"your_logs/your_model_path.pth"

save_path = r"your_save_path"

dcm_path = r"your_test_data"

rm_rf = 'rm -rf '

os.system(rm_rf + save_path)

os.mkdir(save_path)

dcm_list = sorted(os.listdir(dcm_path))

backbone = "resnet50"

num_classes= 2

pretrained = False

model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).to('cuda')

if model_path != '':

    print('Load weights {}.'.format(model_path))

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    load_key, no_load_key, temp_dict = [], [], {}

    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

model.eval()

for i_dcm in dcm_list:

    dcm_info = pydicom.read_file(dcm_path + '/' + i_dcm,force=True)
    dcm = dcm_info.pixel_array

    max_dcm = np.max(dcm)
    min_dcm = np.min(dcm)
    gap = (max_dcm - min_dcm)
    if gap != 0:
        dcm = (dcm - min_dcm) / gap
    else:
        dcm[:] = 0.0
        dcm = dcm/1.0

    dcm = np.tile(dcm,(3,1)).reshape((3,800,800))

    dcm = torch.from_numpy(dcm)

    dcm = torch.unsqueeze(dcm,0).float().to('cuda')
    
    pred = model(dcm)

    pred = torch.max(pred, 1)[1].to('cuda')

    pred[pred ==1 ] = 255

    pred = torch.squeeze(pred,0)

    cv2.imwrite(save_path + '/' + i_dcm[-8:-4] + '.png',pred.cpu().detach().numpy())

    #CUDA_VISIBLE_DEVICES=0 python3 visualize_model.py

