import torch
from torch import nn

# 72 mnist and 98 for cifar10
class ExpandableTinyNet(nn.Module):
    def __init__(self, expansion: int, mnist: bool, colored: bool, out_classes=2):
        super(ExpandableTinyNet, self).__init__()
        fc_inp = 72 if mnist else 98
        channels = 1 if not colored else 3
        self.conv1 = nn.Conv2d(channels, expansion, 3, 1)
        self.conv2 = nn.Conv2d(expansion, 2 * expansion, 3, 2)
        self.fc1 = nn.Linear(fc_inp * expansion, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
class Expandable2LinearReLu(nn.Module):
    def __init__(self, expansion: int, mnist: bool, colored: bool, out_classes=2):
        super(Expandable2LinearReLu, self).__init__()
        channels = 1 if not colored else 3
        inp_img_size = 784 if mnist else 1024
        self.fc1 = nn.Linear(channels*inp_img_size, expansion)
        self.fc2 = nn.Linear(expansion, 5 * expansion)
        self.fc3 = nn.Linear(5 * expansion, out_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
        