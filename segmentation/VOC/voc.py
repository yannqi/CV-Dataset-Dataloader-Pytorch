
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import segmentation.utils.custom_transforms as tr

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

class VOCSegmentation(Dataset):
    def __init__(self, dataset_root, args, split):
        """
        crop_size: (h, w)
        """
        self.split = split
        self.dataset_root = dataset_root
        self.args = args    
        self.id_file = '%s/ImageSets/Segmentation/%s' % (dataset_root, 'trainval.txt' if self.split=='train' else 'test.txt')
        with open(self.id_file, 'r') as f:
            self.img_ids = f.read().split() # 拆分成一个个名字组成list
        self.colormap2label = np.zeros(256**3, dtype=np.uint8) # torch.Size([16777216])
        for i, colormap in enumerate(VOC_COLORMAP):
            # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
            self.colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        self.img_id = self.img_ids[idx]
        _image, mask_image = self._read_voc_images()
        _target = self._voc_label_indices(mask_image)
        _target = Image.fromarray(_target)
        sample = {'image': _image, 'label': _target}
        if self.split == "train":
            return self._transform_train(sample)
        elif self.split == 'val':
            return self._transform_val(sample)
        
    def __len__(self):
        return len(self.img_ids)


    def _read_voc_images(self):
        image= Image.open('%s/JPEGImages/%s.jpg' % (self.dataset_root, self.img_id)).convert("RGB")
        mask_image = Image.open('%s/SegmentationClass/%s.png' % (self.dataset_root, self.img_id)).convert("RGB")
        return image, mask_image # PIL image 0-255
    def _voc_label_indices(self, mask_colormap ):
        mask_colormap = np.array(mask_colormap.convert("RGB")).astype('int32')
        idx = ((mask_colormap[:, :, 0] * 256 + mask_colormap[:, :, 1]) * 256 + mask_colormap[:, :, 2]) 
        return self.colormap2label[idx] # colormap 映射 到colormaplabel中计算的下标


    def _transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.data.BASE_SIZE, crop_size=self.args.data.CROP_SIZE),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def _transform_val(self, sample):
        ori_size = sample['image'].size
        composed_transforms = transforms.Compose([
            
            tr.FixScaleCrop(crop_size = self.args.data.CROP_SIZE),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample), ori_size