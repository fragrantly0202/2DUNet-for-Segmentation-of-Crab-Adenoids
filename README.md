# 2DUNet-for-Segmentation-of-Crab-Adenoids

The repository is a 2DUNet implemented with pytorch, referring to this projects.  I have redesigned the code structure and used the model to perform crab adenoids  segmentation on the private dataset. F1-score can reach 87%.



## Preporcess
Because our crab CT images are dcm files, that is, 16-bit images. The pixels and features of the adenoids are only distributed in a small area. Compared to rbg images with pixel values ranging from 0 to 255, our segmentation target is indeed more difficult.

Our pre-processing operation is to first **calculate the mean and variance of pixel values of the original image corresponding to all adenoid masks**, and take the upper threshold as the mean plus five times the standard deviation, and the lower threshold as the mean minus five times the standard deviation. In this way, we can obtain data with more obvious features after preprocessing

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
            ├── image_1.dcm
            ├── image_2.dcm
            ......
            └── image_x.dcm
        ├── crab_number_2
        ......
        └── crab_number_n
    ├── mask
        ├── crab_number_1
            ├── mask_1.png
            ├── mask_2.png
            ......
            └── mask_x.png
        ├── crab_number_2
        ......
        └── crab_number_n
    └── txt 
        ├── train.txt
        └── val.txt
```
Format your datasets to the above structure and run the `get_txt.py` file to generate the specific txt file.

If you need to run dataset files in other formats, modify the code that reads the data in a file such as `dataloader.py`.

## Train
After preprocessing the dataset and adjusting it to the desired file structure, all you need to do is run ` train.py`

Our code supports multi-card training, you can choose to train on multiple GPUs.

example: `CUDA_VISIBLE_DEVICES=0,1 python3 train.py`

## Result
All of our training logs are stored in the *logs* folder
<div align=center>
  <img width="400" height="400" src= ./imgs/epoch_loss.png/>
  <p class="caption">loss</p>
</div>
<div align=center>
  <img width="400" height="400" src= ./imgs/epoch_f1.png/>
  <p class="caption">f1</p>
</div>
<div align=center>
  <img width="400" height="400" src= ./imgs/epoch_miou.png/>
  <p class="caption">miou</p>
</div>

## Inference
Select the set of crab data you want to get inference and run `visualize_model.py` to get the visualized results.
<div align=center>
  <img width="400" height="400" src= ./imgs/result_1.png/>
  <p class="caption">result 1</p>
</div>
<div align=center>
  <img width="400" height="400" src= ./imgs/result_2.png/>
  <p class="caption">result 2</p>
</div>


