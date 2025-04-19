import torch
import torch.nn as nn
import torchvision.models as models

# Model cho bài toán lane segmentation
class LaneSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.15, freeze_encoder=False, freeze_partial=False):
        super(LaneSegmentationModel, self).__init__()

        # Dùng ResNet18 làm backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Bỏ phần FC layer

        # Freezing toàn bộ encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Hoặc chỉ freeze một phần (layer1 và layer2)
        elif freeze_partial:
            for name, param in self.encoder.named_parameters():
                if "layer1" in name or "layer2" in name:
                    param.requires_grad = False

        # Decoder (U-Net style)
        self.upsample1 = self._upsample_block(512, 256)
        self.upsample2 = self._upsample_block(256, 128)
        self.upsample3 = self._upsample_block(128, 64)
        self.upsample4 = self._upsample_block(64, 32)
        self.upsample5 = self._upsample_block(32, 16)

        # Output layer
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def _upsample_block(self, in_channels, out_channels):
        """Tạo một block upsampling"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder (Upsampling)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        x = self.dropout(x)

        # Output layer
        x = self.final_conv(x)
        return x
        
