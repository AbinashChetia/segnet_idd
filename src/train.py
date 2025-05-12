from segnet_model import SegNet, load_vgg16_bn_weights
from dataset import SegmentationDatasetLite 
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import argparse
import torch.utils.tensorboard as tb
import datetime
import os

IDD_prepared_path = '../data/idd20k_lite_prepared'
NUM_CLASSES = 7
model_output_dir_path = '../trained_models/'

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
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu/cuda/mps)')
    parser.add_argument('--model_output_dir', type=str, default=model_output_dir_path, help='Directory to save the trained model')
    args = parser.parse_args()

    data_dir = args.data_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    patience = args.patience
    device = args.device
    model_output_dir = args.model_output_dir

    num_classes = NUM_CLASSES
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    train_dataset = SegmentationDatasetLite(data_dir=data_dir, transform=data_transforms, mode='train')
    val_dataset = SegmentationDatasetLite(IDD_prepared_path, transform=data_transforms, mode='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SegNet(in_channels=3, num_classes=num_classes).to(device)
    load_vgg16_bn_weights(model) 
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    tb_writer = tb.SummaryWriter(log_dir=f'runs/segnet_training_{timestamp}')
    tb_writer.add_graph(model, next(iter(train_loader))[0].to(device))
    tb_writer.add_text('Hyperparameters', f'Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}, Patience: {patience}')
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
            epoch+1
        )

        print(f"Epoch {epoch+1:3d}/{num_epochs:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    model_name = f'segnet_lite_ep{epoch+1}_{timestamp}.pth'
    torch.save(model.state_dict(), os.path.join(model_output_dir, model_name))
    print(f"Model saved as {model_output_dir}{model_name}.")
    tb_writer.close()
    print("Training complete. TensorBoard logs saved.")