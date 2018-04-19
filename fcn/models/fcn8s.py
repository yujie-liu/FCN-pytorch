# 8x fcn
# To-do: load 16s model
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

class FCN8s(nn.Module):

  
