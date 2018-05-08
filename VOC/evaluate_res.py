## Evaluate the FCN accuracy
#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import fcn
import tqdm
import fcn
import sys
sys.path.insert(0,'../fcn/')
import models
from models.fcn32s_res import FCN32s_RES
from models.resnet import resnet34
from models.resnet import BasicBlock
from models.resnet import Bottleneck
from models.resnet import ResNet
#rom models.fcn32s_res import FCN32s_RES
#from models.fcn16s import FCN16s
#from models.fcn8s import FCN8s
from models.vgg16 import VGGNet
from trainer2 import Trainer
from voc_loader2 import VOCSegmentation

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    model_file = './pth/ResNet-0.pth'
#    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#    model_file = args.model_file

    root = osp.expanduser('../pascal-voc')
    val_loader = torch.utils.data.DataLoader(
        VOCSegmentation(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(val_loader.dataset.CLASSES)
    model = FCN32s_RES(n_class=21)

#    if osp.basename(model_file).startswith('fcn32s'):
#       model = FCN32s(n_class=21, pretrained_net=vgg16)
#    elif osp.basename(model_file).startswith('fcn16s'):
#        model = FCN16s(n_class=21, pretrained_net=vgg16)
#    elif osp.basename(model_file).startswith('fcn8s'):
        # if osp.basename(model_file).startswith('fcn8s-atonce'):
        #     model = FCN8sAtOnce(n_class=21)
        # else:
        #     model = FCN8s(n_class=21, pretrained_net=vgg16)
#        model = FCN8s(n_class=21, pretrained_net=vgg16)
#    else:
#        raise ValueError
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if batch_idx > 20:
            break
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.CLASSES)
                visualizations.append(viz)
    metrics = label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcn.utils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()

