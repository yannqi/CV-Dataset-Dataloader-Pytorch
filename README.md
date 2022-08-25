# CV Dataset Dataloader for Pytorch

# Introduction
This repo is for convenient loading dataloader when create a new project in pytorch style. 
The main Datasets include COCO ,VOC, ImageNet, CIFAR-10, and so on. 
The main methods include Image Detection, Sementatic segmentation.
Keep updating.     

# Use Method
               

# COCO Dataloader

## Data Donwload Save Path
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


