import torch
import numpy as np
from numpy import random
import numbers
import random
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img


class ToTensor(object):
    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            if isinstance(img[i], np.ndarray):
                img[i] = torch.from_numpy(img[i])
            else:
                raise TypeError('Data should be ndarray.')
        return tuple(img)


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *img):
        th, tw = self.size
        h, w = img[0].shape[-2], img[0].shape[-1]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = list(img)
        for i in range(len(img)):
            img[i] = self.crop(img[i], y1, x1, y1 + th, x1 + tw)
        return tuple(img)

    def crop(self, im, x_start, y_start, x_end, y_end):
        if len(im.shape) == 3:
            return im[:, x_start:x_end, y_start:y_end]
        else:
            return im[x_start:x_end, y_start:y_end]


class Normalize(object):
    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            img[i][:2] = (img[i][:2] + 21) / (39 + 21)
            img[i][-1] = (img[i][-1] -186) / (7278 - 186)
        return tuple(img)


class Flip(object):
    def __init__(self, thread = 0.5):
        self.random_thread = thread

    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            assert len(img[i].shape) == 3
            if np.random.random() > self.random_thread:
                img[i] = img[i][:, ::-1, :].copy()
            else:
                img[i] = img[i][:, :, ::-1].copy()
        return tuple(img)


class Transpose(object):
    def __init__(self, thread = 0.5, channel = 2):
        self.random_thread = thread
        self.channel = channel

    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            assert len(img[i].shape) == 3
            if np.random.random() > self.random_thread:
                img[i] = img[i].transpose(0, 2, 1)
        return tuple(img)


class LAPSData(Dataset):
    def __init__(self, transforms=None, timeRange='2017070100,2017110100', root='./dataset/'):
        self.transforms = transforms
        startTime, endTime = timeRange.split(',')
        startTime, endTime = datetime.strptime(startTime, '%Y%m%d%H'), datetime.strptime(endTime, '%Y%m%d%H')
        days = (endTime-startTime).days
        inputList = []
        for hour in range(days*24):
            presentTime = startTime + relativedelta(hours=hour)
            filePath = os.path.join(root, 'InputData', presentTime.strftime('%Y%m%d%H')+'.npy')
            if os.path.exists(filePath):
                inputList.append(filePath)
            else:
                print('%s Not Exist!'%filePath)
        self.fileList = inputList
        self.demFeat = np.load(os.path.join(root, 'DemFeature/dem.npy'))

    def __getitem__(self, index):
        data = np.load(self.fileList[index])
        getData = np.concatenate([data, self.demFeat], axis=0)
        data = self.transforms(*[getData])
        return data[0]

    def __len__(self):
        return len(self.fileList)


if __name__ == '__main__':
    '''-------Test Transform-------'''
    # a = np.arange(2*10*10)
    # a.resize([2, 10, 10])
    # a = a.astype(np.float32)
    # b = a + 1
    #
    # t = Compose([
    #     Normalize(),
    #     Flip(),
    #     Transpose(),
    #     RandomCrop(5),
    #     ToTensor(),
    # ])
    # c, d = t(*[a, b])
    '''-------Test DataLoader-------'''
    import imageio
    t = Compose([
        # RandomCrop(169),
        # Normalize(),
        ToTensor()
    ])
    loader = LAPSData(t, timeRange='2017110100,2018010100')
    loader = DataLoader(loader, batch_size=1, shuffle=False)
    # loader = iter(loader)
    # data = next(loader)
    src = []
    for i, (data) in enumerate(loader):
        feature, dem, label = data[:,:2,::3,::3], data[:,2:], data[:,:2]
        src.append(label[0,0].numpy()+21)
        if (i+1)%24 == 0:
            imageio.mimsave('./temp.gif', src, 'GIF', duration=0.1)
            src = []
            break
        print(data.shape)
    '''--------Show Result-------'''
    # import matplotlib.pyplot as plt
    # plt.figure();plt.imshow(feature.numpy()[0,0])
    # plt.figure();plt.imshow(dem.numpy()[0,0])
    # plt.figure();plt.imshow(label.numpy()[0,0])
