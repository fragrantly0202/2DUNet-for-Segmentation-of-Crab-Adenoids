# coding=utf-8

import os
from centercrop_and_pad import center_crop_and_fill
import pydicom

from pydicom.uid import ExplicitVRLittleEndian
import numpy as np
import cv2 as cv

dcm_path = r"" # your data path
mask_path = r"" # your mask path
save_dcm_path = r"" # your new data path
save_mask_path = r"" # your new mask path

rm_rf = 'rm -rf '

os.system(rm_rf + save_dcm_path)
os.system(rm_rf + save_mask_path)

os.mkdir(save_dcm_path)
os.mkdir(save_mask_path)

crab_numbers = sorted(os.listdir(dcm_path))

crab_len = len(crab_numbers)
crab_mean = 0.0
crab_std = 0.0
for crab_number in crab_numbers:
    crab_number_mean = 0.0
    crab_number_std = 0.0
    effective_cnt = 0

    crab_samples = sorted(os.listdir(dcm_path + '/' + crab_number))

    for crab_sample in crab_samples:
        dcm_info = pydicom.read_file(dcm_path + '/' + crab_number + '/' + crab_sample)
        img = cv.imread(mask_path + '/' + crab_number + '/' + crab_sample[-8:-4] + '.png', cv.COLOR_BGR2GRAY)
        dcm = dcm_info.pixel_array

        if img[img == 255].size != 0:
            effective_cnt += 1

            dcm_copy = dcm

            dcm[img == 0] = 0

            mean_dcm = np.sum(dcm) / dcm[dcm != 0].size

            std_dcm = np.std(dcm[dcm != 0])

            crab_number_mean += mean_dcm
            crab_number_std += std_dcm

    crab_number_mean = int(crab_number_mean / effective_cnt)
    crab_number_std = int(crab_number_std / effective_cnt)

    crab_mean += crab_number_mean
    crab_std += crab_number_std

crab_mean = crab_mean / crab_len
crab_std = crab_std / crab_len

print("mean:", int(crab_mean))
print("std:", int(crab_std))
print("max:", int(crab_mean + 5 * crab_std))
print("min:", int(crab_mean - 5 * crab_std))
print("computer mean and std done!")

target_size = (800, 800)
dcm_target = pydicom.dcmread('your_temp_dcm_file.dcm', force=True) # You have to generate a specific shape of dcm file which will enter into network (example: 800*800)

for crab_number in crab_numbers:

    os.mkdir(save_dcm_path + '/' + crab_number)
    os.mkdir(save_mask_path + '/' + crab_number)

    crab_samples = sorted(os.listdir(dcm_path + '/' + crab_number))

    for crab_sample in crab_samples:
        dcm_info = pydicom.read_file(dcm_path + '/' + crab_number + '/' + crab_sample)  # 读取.dcm文件

        dcm = dcm_info.pixel_array

        dcm[dcm <= int(crab_mean - 5 * crab_std)] = int(crab_mean - 5 * crab_std)
        dcm[dcm >= int(crab_mean + 5 * crab_std)] = int(crab_mean + 5 * crab_std)

        mask = cv.imread(mask_path + '/' + crab_number + '/' + crab_sample[-8:-4] + '.png', cv.COLOR_BGR2GRAY)

        dcm_cropped = center_crop_and_fill(dcm, target_size, fill_value=crab_mean - 5 * crab_std)
        mask_cropped = center_crop_and_fill(mask, target_size, fill_value=0)

        pd = dcm_cropped.tobytes()

        dcm_target.PixelData = pd
        dcm_target.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dcm_target.save_as(save_dcm_path + '/' + crab_number + '/' + crab_sample[-8:])

        cv.imwrite(save_mask_path + '/' + crab_number + '/' + crab_sample[-8:-4] + '.png', mask_cropped)

        print(save_mask_path + '/' + crab_number + '/' + crab_sample[-8:-4] + '.png')


