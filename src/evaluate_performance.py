import argparse
from dataset import SegmentationDataset
from segnet_model import SegNet
import torch
from torchvision import transforms
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
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate SegNet for Semantic Segmentation")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='Number of classes for segmentation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation (cpu/cuda/mps)')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    batch_size = args.batch_size
    num_classes = args.num_classes
    device = args.device

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    val_dataset = SegmentationDataset(data_dir=data_dir, transform=data_transforms, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SegNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path)) 

    ious, miou = evaluate_model(model, val_loader, device)
    
    print(f"Mean IoU: {miou:2.4f}")
    print("Class-wise IoU:")
    for cls in range(num_classes):
        print(f"\t{cls:2d}: {ious[cls]:2.4f}")