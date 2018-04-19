# 16x fcn
# To-do: load 32s model
# 32x fcn
import os.path as osp
import fcn
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class FCN16s(nn.Module):
    def __init__(self, n_class = 21):
        super(FCN16s, self, n_class).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Pool1 = nn.Sequential(*list(vgg_model.features.children())[5])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[6:9])
        self.Pool2 = nn.Sequential(*list(vgg_model.features.children())[10])
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[11:15])
        self.Pool3 = nn.Sequential(*list(vgg_model.features.children())[16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[17:22])
        self.Pool4 = nn.Sequential(*list(vgg_model.features.children())[23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[24:29])
        self.Pool5 = nn.Sequential(*list(vgg_model.features.children())[30])
        # fc6
        fc6 = nn.Conv2d(512, 4096, 7)
        relu6 = nn.ReLU(inplace=True)
        drop6 = nn.Dropout2d()
        self.Conv6 = nn.Sequential(*list([fc6, relu6, drop6]))

        # fc7
        fc7 = nn.Conv2d(4096, 4096, 1)
        relu7 = nn.ReLU(inplace=True)
        drop7 = nn.Dropout2d()

        self.Conv7 = nn.Sequential(*list([fc7, relu7, drop7]))
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)

    def forward(self, x):
        h = x
        h = self.Conv1(h)
        h1 = self.Pool1(h)
        h2 = self.Conv2(h1)
        h2 = self.Pool2(h2)
        h3 = self.Conv3(h2)
        h3 = self.Pool3(h3)
        h4 = self.Conv4(h3)
        h4 = self.Pool4(h4)
        h5 = self.Conv5(h4)
        h5 = self.Pool5(h5)
        h6 = self.Conv6(h5)
        h = self.Conv7(h6)
        h = self.score_fr(h)
        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        return h
