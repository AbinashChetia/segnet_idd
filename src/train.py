from cProfile import label
from segnet_model import SegNet, load_vgg16_bn_weights
from label_hierarchy import LEVEL1, LEVEL2, LEVEL3, LEVEL4
from dataset import SegmentationDataset 
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import argparse
import torch.utils.tensorboard as tb
import datetime
import os

IDD_prepared_path = '../data/idd_segmentation_prepared'
model_output_dir_path = '../trained_models/'

LABEL_MAP_DICT = {
    1: LEVEL1,
    2: LEVEL2,
    3: LEVEL3,
    4: LEVEL4
}

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)               
        masks = masks.to(device).long()           

        optimizer.zero_grad()
        outputs = model(images)             
        loss = criterion(outputs, masks)          

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegNet for Semantic Segmentation")
    parser.add_argument('--data_dir', type=str, default=IDD_prepared_path, help='Path to dataset directory')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3, 4], help='Level of labels to use for segmentation')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu/cuda/mps)')
    parser.add_argument('--model_output_dir', type=str, default=model_output_dir_path, help='Directory to save the trained model')
    args = parser.parse_args()

    data_dir = args.data_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    device = args.device
    model_output_dir = args.model_output_dir
    label_map = LABEL_MAP_DICT[args.level]
    num_classes = len(set(label_map.values()))

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = SegmentationDataset(data_dir=data_dir, transform=data_transforms, mode='train', label_map=label_map)
    val_dataset = SegmentationDataset(IDD_prepared_path, transform=data_transforms, mode='val', label_map=label_map)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SegNet(in_channels=3, num_classes=num_classes).to(device)
    load_vgg16_bn_weights(model)  # Load VGG16-BN weights
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)  # Ignore index 255 for void class
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard setup
    tb_writer = tb.SummaryWriter(log_dir='runs/segnet_training')
    tb_writer.add_graph(model, next(iter(train_loader))[0].to(device))
    tb_writer.add_text('Hyperparameters', f'Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}')
    tb_writer.add_text('Model Summary', str(model))
    tb_writer.add_text('Dataset Summary', f'Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}')
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device
        )
        
        val_loss = evaluate(
            model=model, 
            dataloader=val_loader, 
            criterion=criterion, 
            device=device
        )

        tb_writer.add_scalars(
            'Loss',
            {
                'Train': train_loss,
                'Validation': val_loss
            },
            epoch
        )

        print(f"Epoch {epoch+1:3d}/{num_epochs:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the model 
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    model_name = f'segnet_l{args.level}_e{num_epochs}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    torch.save(model.state_dict(), os.path.join(model_output_dir, model_name))
    print(f"Model saved as {model_name} in {model_output_dir}.")
    tb_writer.close()
    print("Training complete. TensorBoard logs saved.")