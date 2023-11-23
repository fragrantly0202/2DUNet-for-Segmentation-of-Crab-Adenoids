# 2DUNet-for-Segmentation-of-Crab-Adenoids

The repository is a 2DUNet implemented with pytorch, referring to this projects.  I have redesigned the code structure and used the model to perform crab adenoids  segmentation on the private dataset. F1-score can reach 87%.



## Preporcess
Because our crab ct images are dcm files, that is, 16-bit images. The pixels and features of the adenoids are only distributed in a small area. Compared to rbg images with pixel values ranging from 0 to 255, our segmentation target is indeed more difficult.

Our pre-processing operation is to first calculate the mean and variance of pixel values of the original image corresponding to all adenoid masks, and take the upper threshold as the mean plus five times the standard deviation, and the lower threshold as the mean minus five times the standard deviation. In this way, we can obtain data with more obvious features after preprocessing

<div align=center>
  <img width="400" height="400" src= ./imgs/origin_img.png/>
  <p class="caption">origin image</p>
</div>
<div align=center>
  <img width="400" height="400" src= ./imgs/preprocess_img.png/>
  <p class="caption">preprocessed img</p>
</div>

## Code Struture
```angular2
├── nets
    ├── unet.py      # main network
    │── resnet.py      # resnet backbone
    │── vgg.py      # vgg backbone
    └── unet_training.py      # computer metrics and loss, initialization parameter
├── pretrained_model           # store pretrained model
├── logs           # model save path
|── utils            # some related tools
    ├── callbacks.py
    ├── dataloader.py
    ├── utils.py
    └── utils_metrics.py
├── centercrop_and_pad.py  # image transform
├── crab_filter_compute_all.py          # preprocess
├── get_txt.py         # generate txt file
├── visualize_model.py         # generate test images
└── train.py
```

## Dateset Struture
```angular2
└── dataset
    ├── data
        ├── crab_number_1
            ├── image_1
            ├── image_2
            ......
            └── image_x
        ├── crab_number_2
        ......
        └── crab_number_n
    ├── mask
        ├── crab_number_1
            ├── mask_1
            ├── mask_2
            ......
            └── mask_x
        ├── crab_number_2
        ......
        └── crab_number_n
    └── txt 
        ├── train.txt
        └── val.txt
```





