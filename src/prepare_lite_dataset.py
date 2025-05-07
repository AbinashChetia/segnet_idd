import os
import shutil
from tqdm import tqdm

modes = ['seg', 'inst-seg']
MODE = modes[0]

OUTPUT_IMG_TRAIN_DIR = '../data/idd20k_lite_prepared/images/train'
OUTPUT_IMG_VAL_DIR = '../data/idd20k_lite_prepared/images/val'
OUTPUT_IMG_TEST_DIR = '../data/idd20k_lite_prepared/images/test'
OUTPUT_GT_TRAIN_DIR = '../data/idd20k_lite_prepared/labels/train'
OUTPUT_GT_VAL_DIR = '../data/idd20k_lite_prepared/labels/val'

dataset_root_path = '../data/idd20k_lite/'

img_train_path = os.path.join(dataset_root_path, 'leftImg8bit/train')
img_val_path = os.path.join(dataset_root_path, 'leftImg8bit/val')
img_test_path = os.path.join(dataset_root_path, 'leftImg8bit/test')

gt_train_path = os.path.join(dataset_root_path, 'gtFine/train')
gt_val_path = os.path.join(dataset_root_path, 'gtFine/val')

def get_label_path(img_path):
    path_components = img_path.split('/')
    img_name = path_components[-1].split('_')[0]
    dir_name = path_components[-2]
    train_test_val = path_components[-3]

    if train_test_val == 'train' and MODE == 'seg':
        return f'{gt_train_path}/{dir_name}/{img_name}_label.png'

    elif train_test_val == 'train' and MODE == 'inst-seg':
        return f'{gt_train_path}/{dir_name}/{img_name}_inst_label.png'

    elif train_test_val == 'val' and MODE == 'seg':
        return f'{gt_val_path}/{dir_name}/{img_name}_label.png'

    elif train_test_val == 'val' and MODE == 'inst-seg':
        return f'{gt_val_path}/{dir_name}/{img_name}_inst_label.png'
    
    else:
        raise ValueError(f"Invalid path: {img_path}.")
    
train_count = 0
for d in tqdm(os.listdir(img_train_path)):
    if d == '.DS_Store':
        continue
    curr_path = os.path.join(img_train_path, d)
    for i in os.listdir(curr_path):
        if i.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
            continue
        img_path = os.path.join(curr_path, i)
        label_path = get_label_path(img_path)

        if not os.path.exists(label_path):
            print(f"Label path does not exist: {label_path}")
            continue

        if not os.path.exists(OUTPUT_IMG_TRAIN_DIR):
            os.makedirs(OUTPUT_IMG_TRAIN_DIR)
        if not os.path.exists(OUTPUT_GT_TRAIN_DIR):
            os.makedirs(OUTPUT_GT_TRAIN_DIR)

        shutil.copy(img_path, OUTPUT_IMG_TRAIN_DIR)
        shutil.copy(label_path, OUTPUT_GT_TRAIN_DIR)

        train_count += 1

print(f"Copied {train_count} training images and labels.")

val_count = 0
for d in tqdm(os.listdir(img_val_path)):
    if d == '.DS_Store':
        continue
    curr_path = os.path.join(img_val_path, d)
    for i in os.listdir(curr_path):
        if i.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
            continue
        img_path = os.path.join(curr_path, i)
        label_path = get_label_path(img_path)

        if not os.path.exists(label_path):
            print(f"Label path does not exist: {label_path}")
            continue

        if not os.path.exists(OUTPUT_IMG_VAL_DIR):
            os.makedirs(OUTPUT_IMG_VAL_DIR)
        if not os.path.exists(OUTPUT_GT_VAL_DIR):
            os.makedirs(OUTPUT_GT_VAL_DIR)

        shutil.copy(img_path, OUTPUT_IMG_VAL_DIR)
        shutil.copy(label_path, OUTPUT_GT_VAL_DIR)

        val_count += 1

print(f"Copied {val_count} validation images and labels.")

test_count = 0
for d in tqdm(os.listdir(img_test_path)):
    if d == '.DS_Store':
        continue
    curr_path = os.path.join(img_test_path, d)
    for i in os.listdir(curr_path):
        if i.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
            continue
        img_path = os.path.join(curr_path, i)
        if not os.path.exists(OUTPUT_IMG_TEST_DIR):
            os.makedirs(OUTPUT_IMG_TEST_DIR)
        shutil.copy(img_path, OUTPUT_IMG_TEST_DIR)
        
        test_count += 1

print(f"Copied {test_count} test images.")