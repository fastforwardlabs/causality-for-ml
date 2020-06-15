import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

resnet18 = models.resnet18(pretrained=True)

def get_net(name):
    if name == 'WILDCAM':
        # return resnet18_transfer
        return resnet18_extractor

class resnet18_transfer(nn.Module):
    def __init__(self, n_classes=2):
        super(resnet18_transfer, self).__init__()
        image_modules = list(resnet18.children())[:-1]
        self.model = nn.Sequential(*image_modules)
        num_ftrs = resnet18.fc.in_features
        self.fc = nn.Linear(num_ftrs, 1)        

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        e1 = x
        x = self.fc(x)
        return x

    def get_embedding_dim(self):
        return 512

class resnet18_extractor(nn.Module):
    def __init__(self, n_classes=2):
        super(resnet18_extractor, self).__init__()
        image_modules = list(resnet18.children())[:-1]
        self.model = nn.Sequential(*image_modules)
        for param in self.model.parameters():
            param.requires_grad = False
        # newly constructed modules have requires_grad=True by default
        num_ftrs = resnet18.fc.in_features
        self.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        e1 = x
        x = self.fc(x)
        return x

    def get_embedding_dim(self):
        return 512