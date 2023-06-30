#!usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0,bias=False)
        self.features = self.model.fc.in_features
        self.model.fc = nn.Sequential()
        self.lstm = nn.LSTM(self.features, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        
        return x

        
        
        
        
        