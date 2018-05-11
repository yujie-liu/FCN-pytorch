import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
import pytz
import torch
import yaml

import fcn
import sys

sys.path.insert(0, '../fcn/')
import models
from models.fcn_res import FCN32s_RES, FCN8s_RES
from models.resnet import BasicBlock
from models.resnet import Bottleneck
from models.resnet import ResNet
# from models.vgg import VGG
from trainer2 import Trainer
from voc_loader2 import VOCSegmentation
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
configurations = {
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-4,  # originally 1.0e-10
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=100,  # originally 4000
    )
}

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', type=int)
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    resume = (args.resume == 1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    root = osp.expanduser('../pascal-voc')
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        VOCSegmentation(root, split="train", transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        VOCSegmentation(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    start_epoch = 0
    start_iteration = 0
    model = FCN8s_RES(n_class=21, pretrained=True)
    if resume:
        model = FCN8s_RES(n_class=21)
        checkpoint = torch.load("./pth/ResNet8s.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    if cuda:
        model = model.cuda()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
