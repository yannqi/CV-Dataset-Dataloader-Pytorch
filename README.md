# CV Dataset Dataloader for Pytorch

# Introduction
This repo is for convenient loading dataloader when create a new project in pytorch style. 
The main Datasets include COCO ,VOC, ImageNet, CIFAR-10, and so on. 
The main methods include Object Detection, Sementatic segmentation.
Keep updating.     

**Image Classification**
- [x] ImageNet 1k
    Download by Linux. [linux下载/解压ImageNet-1k数据集](https://blog.csdn.net/qq_45588019/article/details/125642466)

**Object Detection:**

- [x] COCO 
- [ ] VOC
- [x] Data Augmentation

**Sementatic Segmentation:**
- [x] COCO
- [x] VOC               
- [x] Data Augmentation

**Others**
- [ ] Data with DALI

# COCO Dataloader
include : Object Detection, Sementatic segmentation. 

(With data Aug.)
## COCO Data Donwload Save Path
### Root Directory 
    └──  COCO 
        ├──images
            ├── train2017: All train images(118287张)
            ├── val2017: All validate images(5000张)
        ├── annotations
            ├── instances_train2017.json
            ├── instances_val2017.json
            ├── captions_train2017.json
            ├── captions_val2017.json
            ├── person_keypoints_train2017.json
            └── person_keypoints_val2017.json
        └── coco_labels.txt
- instances_train2017.json:    
    Training set annotation file corresponding to object detection and semantic 
- instances_val2017.json: 
    Validation set annotation file corresponding to object detection and semantic 


## VOC Dataloader

## VOC Data Donwload Save Path

### Root Directory 
Actually, you only need to unzip/tar the download dataset, the dataset path will unzip as below:

    └──  VOC2012
        ├──SegmentationObject
        ├──SegmentationClass   
        ├──JPEGImages
        ├──ImageSets
        └──Annotations