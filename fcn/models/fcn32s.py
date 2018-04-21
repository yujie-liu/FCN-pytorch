
import os.path as osp
import fcn
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

#
# class FCN32s(nn.Module):
#     def __init__(self, n_class = 21):
#         super(FCN32s, self, n_class).__init__()
#         vgg_model = torchvision.models.vgg16(pretrained=True)
#         self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
#         self.Pool1 = nn.Sequential(*list(vgg_model.features.children())[5])
#         self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[6:9])
#         self.Pool2 = nn.Sequential(*list(vgg_model.features.children())[10])
#         self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[11:15])
#         self.Pool3 = nn.Sequential(*list(vgg_model.features.children())[16])
#         self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[17:22])
#         self.Pool4 = nn.Sequential(*list(vgg_model.features.children())[23])
#         self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[24:29])
#         self.Pool5 = nn.Sequential(*list(vgg_model.features.children())[30])
#         # fc6
#         fc6 = nn.Conv2d(512, 4096, 7)
#         relu6 = nn.ReLU(inplace=True)
#         drop6 = nn.Dropout2d()
#         self.Conv6 = nn.Sequential(*list([fc6, relu6, drop6]))
#
#         # fc7
#         fc7 = nn.Conv2d(4096, 4096, 1)
#         relu7 = nn.ReLU(inplace=True)
#         drop7 = nn.Dropout2d()
#
#         self.Conv7 = nn.Sequential(*list([fc7, relu7, drop7]))
#         self.score_fr = nn.Conv2d(4096, n_class, 1)
#         self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
#         # Need to initialize conv6 and conv7 with bilinear interpolation
#
#     # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
#     def get_upsampling_weight(in_channels, out_channels, kernel_size):
#         """Make a 2D bilinear kernel suitable for upsampling"""
#         factor = (kernel_size + 1) // 2
#         if kernel_size % 2 == 1:
#             center = factor - 1
#         else:
#             center = factor - 0.5
#         og = np.ogrid[:kernel_size, :kernel_size]
#         filt = (1 - abs(og[0] - center) / factor) * \
#                (1 - abs(og[1] - center) / factor)
#         weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
#                           dtype=np.float64)
#         weight[range(in_channels), range(out_channels), :, :] = filt
#         return torch.from_numpy(weight).float()
#
#     #def copy_params:
#
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.zero_()
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             if isinstance(m, nn.ConvTranspose2d):
#                 assert m.kernel_size[0] == m.kernel_size[1]
#                 initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
#                 m.weight.data.copy_(initial_weight)
#
#     def forward(self, x):
#         h = x
#         h = self.Conv1(h)
#         h1 = self.Pool1(h)
#         h2 = self.Conv2(h1)
#         h2 = self.Pool2(h2)
#         h3 = self.Conv3(h2)
#         h3 = self.Pool3(h3)
#         h4 = self.Conv4(h3)
#         h4 = self.Pool4(h4)
#         h5 = self.Conv5(h4)
#         h5 = self.Pool5(h5)
#         h6 = self.Conv6(h5)
#         h = self.Conv7(h6)
#         h = self.score_fr(h)
#         h = self.upscore(h)
#         h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
#         return h
#
#
# if __name__ == '__main__':
#     testnet = FCN32s()
#     for child in testnet.children():
#         for param in child.parameters():
#             print(param)
#             break
#         break

class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super(FCN32s, self, pretrained_net, n_class).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
