# create dataset class for loading and processing data from data folder
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode should be one of ['train', 'val', 'test']")
        
        if mode == 'train':
            self.images = sorted(os.listdir(os.path.join(data_dir, 'images/train')))
            self.labels = sorted(os.listdir(os.path.join(data_dir, 'labels/train')))

        elif mode == 'val':
            self.images = sorted(os.listdir(os.path.join(data_dir, 'images/val')))
            self.labels = sorted(os.listdir(os.path.join(data_dir, 'labels/val')))

        elif mode == 'test':
            self.images = sorted(os.listdir(os.path.join(data_dir, 'images/test')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode == 'val':
            img_path = os.path.join(self.data_dir, 'images', self.mode, self.images[idx])
            label_path = os.path.join(self.data_dir, 'labels', self.mode, self.labels[idx])

            if not os.path.exists(img_path) or not os.path.exists(label_path):
                raise FileNotFoundError(f"Image or label file not found: {img_path} or {label_path}")
        
        else:
            img_path = os.path.join(self.data_dir, 'images', self.mode, self.images[idx])
            label_path = None
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L') if label_path else None

        if self.transform:
            image = self.transform(image)
            if label is not None:
                label = label.resize((image.shape[2], image.shape[1]), resample=Image.NEAREST)  # ensure same H, W
                label = np.array(label, dtype=np.uint8)
                label = torch.from_numpy(label).long()

        return image, label