import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

NUM_CLASSES = 34
IGNORE_INDEX = 255

def compute_iou(conf_matrix):
    ious = []
    for cls in range(NUM_CLASSES):
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            ious.append(float('nan')) 
        else:
            ious.append(tp / denom)
    return ious

def evaluate_model(model, dataloader, device):
    model.eval()
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Loop over batch
            for pred, gt in zip(preds, labels):
                mask = gt != IGNORE_INDEX  # Mask to ignore unlabeled regions
                pred = pred[mask]
                gt = gt[mask]

                # Flatten and convert to numpy
                pred_np = pred.cpu().numpy().flatten()
                gt_np = gt.cpu().numpy().flatten()

                # Update confusion matrix
                conf_matrix += confusion_matrix(gt_np, pred_np, labels=list(range(NUM_CLASSES)))

    ious = compute_iou(conf_matrix)
    miou = np.nanmean(ious)
    return ious, miou