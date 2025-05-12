import label_hierarchy as label_maps

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', label_map=label_maps.LEVEL1):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.label_map = label_map
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
            self.labels = sorted(os.listdir(os.path.join(data_dir, 'labels/test')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_dir, 'images', self.mode, self.images[idx])
        label_path = os.path.join(self.data_dir, 'labels', self.mode, self.labels[idx])

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Image or label file not found: {img_path} or {label_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = Image.fromarray(generate_mask_from_json(label_path, self.label_map)) if label_path else None

        if self.transform:
            image = self.transform(image)
            if label is not None:
                label = label.resize((image.shape[2], image.shape[1]), resample=Image.Resampling.NEAREST)
                label = np.array(label, dtype=np.uint8)
                label = torch.from_numpy(label).long()

        return image, label

def generate_mask_from_json(json_path, label_map):
    with open(json_path, 'r') as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros((height, width), dtype=np.uint8)
    mask.fill(label_map['out of roi']) 

    for obj in data["objects"]:
        label = obj["label"]
        polygon = obj["polygon"]
        if label not in label_map:
            continue
        class_id = label_map[label]

        pts = np.array(polygon, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [pts], color=class_id)

    return mask

class SegmentationDatasetLite(Dataset):
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
            self.labels = sorted(os.listdir(os.path.join(data_dir, 'labels/test')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_dir, 'images', self.mode, self.images[idx])
        label_path = os.path.join(self.data_dir, 'labels', self.mode, self.labels[idx])

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Image or label file not found: {img_path} or {label_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L') if label_path else None

        if self.transform:
            image = self.transform(image)
            if label is not None:
                label = label.resize((image.shape[2], image.shape[1]), resample=Image.Resampling.NEAREST)
                label = np.array(label, dtype=np.uint8)
                label = torch.from_numpy(label).long()

        return image, label