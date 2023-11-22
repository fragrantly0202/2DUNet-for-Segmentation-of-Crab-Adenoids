# 2DUNet-for-Segmentation-of-Crab-Adenoids

The repository is a 2DUNet implemented with pytorch, referring to this projects.  I have redesigned the code structure and used the model to perform crab adenoids  segmentation on the private dataset. F1-score can reach 87%.



# Preporcessing
Because our crab ct images are dcm files, that is, 16-bit images. The pixels and features of the adenoids are only distributed in a small area. Compared to rbg images with pixel values ranging from 0 to 255, our segmentation target is indeed more difficult.

Our pre-processing operation is to first calculate the mean and variance of pixel values of the original image corresponding to all adenoid masks, and take the upper threshold as the mean plus five times the standard deviation, and the lower threshold as the mean minus five times the standard deviation. In this way, we can obtain data with more obvious features after preprocessing
