from __future__ import print_function

import errno
import argparse
import os
import sys
import tarfile
import torch
import torch.utils.data as data
from PIL import Image
from six.moves import urllib
import numpy as np
import scipy
import torchvision


class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    BASE_DIR = 'VOCdevkit/VOC2012'
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

    def __init__(self,
                 root,
                 split="train",
                 transform=False,
                 target_transform=False,
                 download=False):
        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self._transform = transform
        self.target_transform = target_transform
        self.split = split

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.'
                               ' You can use download=True to download it')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        _split_f = os.path.join(_splits_dir, 'train.txt')
        if not self.split == "train":
            _split_f = os.path.join(_splits_dir, 'val.txt')  # trainval.txt

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index])
        _target = Image.open(self.masks[index])
        if _img.size[0] <= 224:
            _img = _img.resize((225, _img.size[1]), Image.NEAREST)
            _target = _target.resize((225, _target.size[1]), Image.NEAREST)
        if _img.size[1] <= 224:
            temp = _img.size[0]
            _img = _img.resize((temp, 225), Image.NEAREST)
            _target = _target.resize((temp, 225), Image.NEAREST)

        _img = np.array(_img, dtype=np.uint8)
        _target = np.array(_target, dtype=np.int32)
        _target[_target == 255] = -1
        if self._transform:
            return self.transform(_img, _target)

        return _img, _target

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        return True

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('File extracted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', type=int, default=0)
    args = parser.parse_args()
    download = args.download
    pascal = VOCSegmentation('../pascal-voc', download)
