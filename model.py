from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution (DSConv) module.
    This module performs a depthwise convolution followed by a pointwise convolution,
    providing a lightweight alternative to standard convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the DSConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DSConv, self).__init__()
        # Depthwise convolution: applies a filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        # Pointwise convolution: combines depthwise outputs into out_channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Batch normalization to stabilize training
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU activation for non-linearity
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DSConv module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W).
        """
        # Sequentially apply depthwise conv, pointwise conv, batch norm, and ReLU
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class MobileNetV2Encoder(nn.Module):
    """
    Encoder module using MobileNetV2.
    Extracts features from the input image and provides skip connections for the decoder.
    """
    def __init__(self) -> None:
        """
        Initialize the MobileNetV2Encoder module.
        """
        super(MobileNetV2Encoder, self).__init__()
        # Load pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        # Use the feature extraction layers (19 layers in total)
        self.features = mobilenet.features  # Sequential of 19 layers (0 to 18)
    
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the MobileNetV2Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            tuple[list[torch.Tensor], torch.Tensor]: A tuple containing:
                - skips: List of tensors for skip connections.
                - x: Final output tensor from the encoder.
        """
        skips = []
        # Process input through each layer of MobileNetV2 features
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Store outputs at specific layers for skip connections
            if i in [1, 3, 6, 13]:  # Layers before downsampling steps
                skips.append(x)
        # Return skip connections and bottleneck output
        return skips, x


class DecoderBlock(nn.Module):
    """
    Decoder block for the U-Net architecture.
    Combines upsampling, skip connections, and convolutions to refine features.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """
        Initialize the DecoderBlock.

        Args:
            in_channels (int): Number of input channels from the previous layer.
            skip_channels (int): Number of channels in the skip connection.
            out_channels (int): Number of output channels after convolutions.
        """
        super(DecoderBlock, self).__init__()
        # Upsampling to double the spatial dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # First DSConv after concatenating upsampled input and skip connection
        self.conv1 = DSConv(in_channels + skip_channels, out_channels)
        # Second DSConv to further refine the features
        self.conv2 = DSConv(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor from the previous layer.
            skip (torch.Tensor): Skip connection tensor from the encoder.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        # Upsample input, concatenate with skip connection, and apply convolutions
        return self.conv2(self.conv1(torch.cat([self.upsample(x), skip], dim=1)))


class UNETMobileNetV2(nn.Module):
    """
    U-Net model with MobileNetV2 as the encoder for semantic segmentation.
    Combines an efficient encoder with a decoder that uses skip connections to produce detailed segmentation maps.
    """
    def __init__(self, num_classes: int) -> None:
        """
        Initialize the UNETMobileNetV2 model.

        Args:
            num_classes (int): Number of segmentation classes.
        """
        super(UNETMobileNetV2, self).__init__()
        self.num_classes = num_classes
        # Encoder: Feature extraction using MobileNetV2
        self.encoder = MobileNetV2Encoder()
        
        # Bottleneck: Process the encoder output with additional convolutions
        self.bottleneck = nn.Sequential(
            DSConv(1280, 1024),  # Reduce channels from 1280 to 1024
            DSConv(1024, 1024)   # Further refine features
        )
        
        # Decoder Blocks: Upsample and integrate skip connections
        self.dec4 = DecoderBlock(1024, 96, 512)  # From 7x7 to 14x14
        self.dec3 = DecoderBlock(512, 32, 256)   # From 14x14 to 28x28
        self.dec2 = DecoderBlock(256, 24, 128)   # From 28x28 to 56x56
        self.dec1 = DecoderBlock(128, 16, 64)    # From 56x56 to 112x112
        
        # Final layers: Upsample to original size and produce segmentation map
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Loss function: CrossEntropyLoss for multi-class segmentation
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the UNETMobileNetV2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
            target (torch.Tensor, optional): Ground truth labels of shape (batch_size, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - pred: Predicted segmentation map of shape (batch_size, num_classes, H, W).
                - loss: Loss value if target is provided, otherwise None.
        """
        # Encoder: Extract features and skip connections
        skips, x = self.encoder(x)
        # skips: [0: 112x112,16], [1: 56x56,24], [2: 28x28,32], [3: 14x14,96]
        # x: [batch_size, 1280, H/32, W/32] (e.g., 7x7 for 224x224 input)
        
        # Bottleneck: Refine the deepest features
        x = self.bottleneck(x)
        
        # Decoder: Progressively upsample and combine with skip connections
        x = self.dec4(x, skips[3])  # 7x7 -> 14x14
        x = self.dec3(x, skips[2])  # 14x14 -> 28x28
        x = self.dec2(x, skips[1])  # 28x28 -> 56x56
        x = self.dec1(x, skips[0])  # 56x56 -> 112x112
        
        # Final prediction: Upsample to input size and apply final convolution
        pred = self.final_conv(self.final_upsample(x))
        
        # Compute loss if target is provided
        if target is not None:
            loss = self.combined_loss(pred, target)
            return pred, loss
        return pred, None
    
    def combined_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.4, beta: float = 0.6) -> torch.Tensor:
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
        # Calculate CrossEntropyLoss
        ce_loss = self.ce_loss(pred, target)
        # Calculate Dice Loss
        dice = self.dice_loss(pred, target)
        # Combine losses with specified weights
        return alpha * ce_loss + beta * dice
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
        """
        Compute the Dice Loss for multi-class segmentation.

        Args:
            pred (torch.Tensor): Predicted logits of shape (batch_size, num_classes, H, W).
            target (torch.Tensor): Ground truth labels of shape (batch_size, H, W).
            smooth (float): Smoothing factor to avoid division by zero (default: 1e-8).

        Returns:
            torch.Tensor: Scalar Dice Loss value.
        """
        # Convert logits to probabilities
        pred = torch.softmax(pred, dim=1)
        
        # Convert target to one-hot format matching pred's shape
        target_one_hot = torch.nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient for each class
        dice = 0
        for c in range(self.num_classes):
            pred_c = pred[:, c, :, :]  # Predictions for class c
            target_c = target_one_hot[:, c, :, :]  # Ground truth for class c
            intersection = (pred_c * target_c).sum()  # Overlap between prediction and target
            union = pred_c.sum() + target_c.sum()  # Total area of prediction and target
            # Dice coefficient for class c
            dice += (2. * intersection + smooth) / (union + smooth)
        
        # Average Dice over all classes and convert to loss
        return 1 - (dice / self.num_classes)
    
    def optimizer(self) -> torch.optim.Optimizer:
        """
        Provide an optimizer for training the model.

        Args:
            *args: Additional positional arguments for the optimizer.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            torch.optim.Optimizer: An instance of the AdamW optimizer configured with model parameters.
        """
        # Return AdamW optimizer with model parameters
        return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay= 1e-4)