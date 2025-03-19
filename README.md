# ade20k_semantic_segmentation
Implementing Models for ADE20K Dataset Segmentation

1. Trained Using MobileNetV2 - The Model is not strong enough to predict 151 classes(need to try other models)


```
# to train the model
from dataset import ADE20KDATASET
from model import UNETMobileNetV2
from train import train_model
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# Main Execution
if __name__ == "__main__":
    NUM_CLASSES = 151
    BATCH_SIZE = 36
    NUM_EPOCHS = 20
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_dataset = ADE20KDATASET(split='train')
    val_dataset = ADE20KDATASET(split='val')
    test_dataset = ADE20KDATASET(split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize Model
    model = UNETMobileNetV2(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    # Train the Model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path='train.pth'
    )
```