from matplotlib.colorbar import make_axes_gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from mx import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Linear, simd_add
from mx.simd_ops import simd_split

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride, mx_specs):
        super(Bottleneck, self).__init__()
        
        self.mx_specs = mx_specs
        
        self.residual = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            Conv2d(in_channels, out_channels, kernel_size=1, bias=False, mx_specs=mx_specs),
            # nn.BatchNorm2d(out_channels),
            BatchNorm2d(out_channels, mx_specs),
            # nn.ReLU(inplace=True),
            ReLU(inplace=True, mx_specs=mx_specs),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, mx_specs=mx_specs),
            # nn.BatchNorm2d(out_channels),
            BatchNorm2d(out_channels, mx_specs=mx_specs),
            # nn.ReLU(out_channels),
            ReLU(inplace=True, mx_specs=mx_specs),
            # nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False, mx_specs=mx_specs),
            # nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            BatchNorm2d(out_channels * Bottleneck.expansion, mx_specs=mx_specs)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, padding=1, bias=False)
                Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, padding=1, bias=False, mx_specs=mx_specs),
                # nn.BatchNorm2d(out_channels * Bottleneck.expansion),
                BatchNorm2d(out_channels * Bottleneck.expansion, mx_specs=mx_specs)
            )
            
    def forward(self, x):
        x, residual = simd_split(x, mx_specs=self.mx_specs)
        
        bottleneck_output = self.residual(x)
        output = simd_add(residual, bottleneck_output, mx_specs=self.mx_specs)
        output = ReLU(inplace=True, mx_specs=self.mx_specs)
        return output
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, mx_specs=None):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.mx_specs = mx_specs
        
        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            Conv2d(3, 64, kernel_size=3, padding=1, bias=False, mx_specs=mx_specs),
            # nn.BatchNorm2d(64),
            BatchNorm2d(64, mx_specs=mx_specs),
            # nn.ReLU(inplace=True)
            ReLU(inplace=True, mx_specs=mx_specs)
        )
        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = AdaptiveAvgPool2d((1, 1), mx_specs=mx_specs)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = Linear(512 * block.expansion, num_classes, mx_specs=mx_specs)
        
        
    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.mx_specs))
            self.in_channels *= block.expansion
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
def resnet152(mx_specs=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], mx_specs=mx_specs)

def resnet50(mx_specs=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], mx_specs=mx_specs)
