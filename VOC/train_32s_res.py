import argparse
import datetime
import os
import os.path as osp
import shlex
import pytz
import torch
import yaml
import fcn
import sys

sys.path.insert(0, '../fcn/')
import models
from models.fcn_res import FCN32s_RES, FCN8s_RES
from trainer2 import Trainer
from voc_loader2 import VOCSegmentation
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
configurations = {
    1: dict(
        max_iteration=100000,
        lr=1.0e-4,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=1000,  # originally 4000
    )
}


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        BasicBlock,
        Bottleneck,
        nn.BatchNorm2d,
        nn.AvgPool2d,
        nn.Linear,
        ResNet
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

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

    # 2. model

    start_epoch = 0
    start_iteration = 0
    model = FCN32s_RES(n_class=21, pretrained=True)
    if resume:
        model = FCN32s_RES(n_class=21)
        checkpoint = torch.load("./pth/ResNet32s-2.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    if cuda:
        model = model.cuda()

    #param = get_parameters(model, bias=False)
    #param1 = [x for x in param if x.requires_grad]
    #param2 = get_parameters(model, bias=True)
    #param2 = [x for x in param if x is not None and x.requires_grad]
    optim = torch.optim.SGD(
        model.parameters(),
	#[
            #{'params': param1},
            #{'params': param2,
            # 'lr': cfg['lr'] * 2, 'weight_decay': 0},
        #],
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

