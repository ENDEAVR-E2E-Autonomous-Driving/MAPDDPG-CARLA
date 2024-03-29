import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

"""
Channel Attention
"""
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16) -> None:
        super(ChannelAttention, self).__init__()
        # reduction_ratio reduces number of channels in the hidden layer of the MLP to decrease number of parameters
        # apply max and average pooling to the spatial dimensions
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # shared MLP (multi-layer perceptron)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        maxpooled = self.maxpool(x)
        avgpooled = self.avgpool(x)

        # flatten before passing through MLP
        maxpooled = maxpooled.view(x.size(0), -1)
        avgpooled = avgpooled.view(x.size(0), -1)

        # pass through MLP
        max_out = self.shared_mlp(maxpooled)
        avg_out = self.shared_mlp(avgpooled)

        # Concatenate channel-wise features
        out = max_out + avg_out
        out = out.view(x.size(0), -1, 1, 1) # reshape back to the original size but with spatial dims 1x1

        attention = self.sigmoid(out)

        return attention
    

"""
Spatial Attention
Consider modifying later
"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7) -> None:
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # average and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv(x)

        return self.sigmoid(x)
    

"""
Convolutional Block Attention Mechanism
"""
class CBAM(nn.Module):
    def __init__(self, in_channels) -> None:
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, f):
        channel_attention = self.channel_attention.forward(f)
        f_channel = torch.matmul(channel_attention, f)

        spatial_attention = self.spatial_attention.forward(f_channel)
        cbam_output = torch.matmul(spatial_attention, f_channel)

        return cbam_output

