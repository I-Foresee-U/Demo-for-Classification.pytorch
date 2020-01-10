'''
Model construction

author Gu Jiapan
'''

import torch
import torch.nn as nn

from ResNet import resnet18


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet1 = resnet18(input_channels=2)
        self.resnet2 = resnet18(input_channels=2)
        self.conv1 = nn.Conv2d(3*512, 512, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 100)


    def forward(self, x):
        x1 = torch.cat((torch.unsqueeze(x[:, 0, :, :], 1),
                        torch.unsqueeze(x[:, 2, :, :], 1),
                        ), 1)
        x3 = torch.cat((torch.unsqueeze(x[:, 1, :, :], 1),
                        torch.unsqueeze(x[:, 3, :, :], 1),
                        ), 1)
        fm1 = self.resnet1(x1)
        fm2 = self.resnet2(x2)
        fm = torch.cat((fm1,fm2),1)
        out = self.conv1(fm)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
