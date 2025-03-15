from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models


class DSConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class MobileNetV2Encoder(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2Encoder, self).__init__()
        mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        self.features = mobilenet.features  # Sequential of 19 layers (0 to 18)
    
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [1, 3, 6, 13]:  # Layers before downsampling steps
                skips.append(x)
        # skipconnections and bottleneck 
        return skips, x


# Decoder Block: Upsampling + Skip Connection + Convolutions
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DSConv(in_channels + skip_channels, out_channels)
        self.conv2 = DSConv(out_channels, out_channels)
    
    def forward(self, x, skip) -> torch.Tensor:
        return self.conv2(self.conv1(torch.cat([self.upsample(x), skip], dim=1)))


# Full U-Net Model
class UNETMobileNetV2(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(UNETMobileNetV2, self).__init__()
        self.num_classes = num_classes
        # Encoder
        self.encoder = MobileNetV2Encoder()
        
        # Bottleneck: Additional Conv Layers
        self.bottleneck = nn.Sequential(
            DSConv(1280, 1024),  # Reduce from 1280 channels, option to increase if desired
            DSConv(1024, 1024)
        )
        
        # Decoder Blocks
        self.dec4 = DecoderBlock(1024, 96, 512)  # 7x7 -> 14x14
        self.dec3 = DecoderBlock(512, 32, 256)   # 14x14 -> 28x28
        self.dec2 = DecoderBlock(256, 24, 128)   # 28x28 -> 56x56
        self.dec1 = DecoderBlock(128, 16, 64)    # 56x56 -> 112x112
        
        # Final Upsampling and Output Layer
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Loss Functions
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips, x = self.encoder(x)
        # skips: [0: 112x112,16], [1: 56x56,24], [2: 28x28,32], [3: 14x14,96]
        # x: [batch_size, 1280, H/32, W/32] (e.g., 7x7 for 224x224 input)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with Skip Connections
        x = self.dec4(x, skips[3])  # 7x7 -> 14x14
        x = self.dec3(x, skips[2])  # 14x14 -> 28x28
        x = self.dec2(x, skips[1])  # 28x28 -> 56x56
        x = self.dec1(x, skips[0])  # 56x56 -> 112x112

        return self.final_conv(self.final_upsample(x))
    
    def combined_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
        """
        Compute the combined CrossEntropyLoss and Dice Loss.

        Args:
            pred (torch.Tensor): Predicted logits of shape (batch_size, num_classes, H, W).
            target (torch.Tensor): Ground truth labels of shape (batch_size, H, W).
            alpha (float): Weight for CrossEntropyLoss (default: 1.0).
            beta (float): Weight for Dice Loss (default: 1.0).

        Returns:
            torch.Tensor: Combined loss value.
        """
        ce_loss = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return alpha * ce_loss + beta * dice
    
    @staticmethod
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Compute the Dice Loss for multi-class segmentation.

        Args:
            pred (torch.Tensor): Predicted logits of shape (batch_size, num_classes, H, W).
            target (torch.Tensor): Ground truth labels of shape (batch_size, H, W).
            num_classes (int): Number of classes in the segmentation task.
            smooth (float): Smoothing factor to prevent division by zero (default: 1e-6).

        Returns:
            torch.Tensor: Scalar Dice Loss value.
        """
        # Convert logits to probabilities
        pred = torch.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding and match the prediction shape
        target_one_hot = torch.nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Dice Loss for each class and average
        dice = 0
        for c in range(self.num_classes):
            pred_c = pred[:, c, :, :]  # Predictions for class c
            target_c = target_one_hot[:, c, :, :]  # Ground truth for class c
            intersection = (pred_c * target_c).sum()  # Overlap
            union = pred_c.sum() + target_c.sum()  # Total area
            dice += (2. * intersection + smooth) / (union + smooth)
        
        # Average over classes and compute loss
        return 1 - (dice / self.num_classes)
    
    def optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), *args, **kwargs)