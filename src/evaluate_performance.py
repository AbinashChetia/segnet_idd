from dataset import SegmentationDatasetLite
from segnet_model import SegNet
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse

IDD_prepared_path = '../data/idd20k_lite_prepared'
NUM_CLASSES = 7

def compute_iou(conf_matrix, labels):
    ious = {}
    for cls in labels:
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            ious[cls] = np.nan
        else:
            ious[cls] = tp / denom
    return ious

def evaluate_model(model, dataloader, device):
    model.eval()
    num_classes = NUM_CLASSES
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Loop over batch
            for pred, gt in zip(preds, labels):
                mask = gt != 255
                pred = pred[mask]
                gt = gt[mask]

                pred_np = pred.cpu().numpy().flatten()
                gt_np = gt.cpu().numpy().flatten()

                conf_matrix += confusion_matrix(gt_np, pred_np, labels=list(range(num_classes)))

    ious = compute_iou(conf_matrix, list(range(num_classes)))
    miou = np.nanmean(list(ious.values()))
    return ious, miou, conf_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate SegNet for Semantic Segmentation")
    parser.add_argument('--data_dir', type=str, default=IDD_prepared_path, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation (cpu/cuda/mps)')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    batch_size = args.batch_size
    device = args.device
    num_classes = NUM_CLASSES

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = SegmentationDatasetLite(data_dir=data_dir, transform=data_transforms, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SegNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path)) 

    ious, miou, _ = evaluate_model(model, test_loader, device)
    
    print(f"Mean IoU: {miou:2.4f}")
    print("Class-wise IoU:")
    for k, v in ious.items():
        print(f"\t{k:2d}: {v:2.4f}")