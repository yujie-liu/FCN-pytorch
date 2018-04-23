from __future__ import print_function

import errno
import hashlib
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
    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    FILE = "VOCtrainval_06-Nov-2007.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2007'
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        #print(img.shape)
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
            _split_f = os.path.join(_splits_dir, 'val.txt') # trainval.txt

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
        print(index)
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
        # mat = scipy.io.loadmat(self.masks[index])
        # _target = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        # _target[_target == 255] = -1
        if self._transform:
            print("transform was not none")
            return self.transform(_img, _target)

        # todo(bdd) : perhaps transformations should be applied differently to masks?
        # if self.target_transform:
        #     _target = self.untransform(_target)

        return _img, _target

    # data_file = self.files[self.split][index]
    # # load image
    # img_file = data_file['img']
    # img = PIL.Image.open(img_file)
    # img = np.array(img, dtype=np.uint8)
    # # load label
    # lbl_file = data_file['lbl']
    # mat = scipy.io.loadmat(lbl_file)
    # lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
    # lbl[lbl == 255] = -1
    # if self._transform:
    #     return self.transform(img, lbl)
    # else:
    #     return img, lbl

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        # md5c = hashlib.md5(open(_fpath, 'rb').read()).hexdigest()
        # if _md5c != self.MD5:
        #     print(" MD5({}) did not match MD5({}) expected for {}".format(
        #         _md5c, self.MD5, _fpath))
        #     return False
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

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


if __name__ == '__main__':
    pascal = VOCSegmentation('../pascal-voc', download=False)
    print(pascal[3])
