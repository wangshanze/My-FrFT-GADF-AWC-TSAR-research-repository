import torch
import torch.nn as nn

# Basic Block for ResNet (weakened version)
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # weakened convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Tanh is MUCH weaker than ReLU
        self.act = nn.Tanh()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Add dropout inside block → very effective degradation
        self.dropout = nn.Dropout(0.30)

        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)      # degrade feature learning

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.act(out)
        out = self.dropout(out)      # degrade residual output
        
        return out



# 4-layer simplified ResNet (weakened version)
class SimpleResNet(nn.Module):
    def __init__(self, num_classes: int, input_channels: int, dropout_p: float = 0.1):
        super().__init__()

        # reduce conv1 channels to 32 (weaker)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Reduce channel sizes significantly
        self.layer1 = self._make_layer(32, 16, blocks=1, stride=1)
        self.layer2 = self._make_layer(16, 32, blocks=1, stride=2)
        self.layer3 = self._make_layer(32, 64, blocks=1, stride=2)
        self.layer4 = self._make_layer(64, 96, blocks=1, stride=2)

        # Add dropout between layers to weaken network
        self.inter_dropout = nn.Dropout(0.35)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # classifier
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(96 * BasicBlock.expansion, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.40),   # degrade skip connection
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        return nn.Sequential(*layers)
    
    def forward(self, x):

        # add slight noise during training
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.inter_dropout(x)

        x = self.layer2(x)
        x = self.inter_dropout(x)

        x = self.layer3(x)
        x = self.inter_dropout(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x



class FrFTGADFResNet(nn.Module):
    def __init__(self, num_classes: int, input_channels: int, dropout_p: float = 0.8):
        super().__init__()
        self.backbone = SimpleResNet(num_classes, input_channels, dropout_p)

    def forward(self, x):
        return self.backbone(x)

