import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt
from lib.config import edict as edict
import os
import shutil


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * ((1 - float(epoch)/args.nEpochs)**args.power)
    for param in optimizer.param_groups:
        param['lr'] = lr


def saveCheckpoint(epoch, state, comment, isBest=False):
    '''Save at each epoch and copy at best epoch'''
    cpkRoot = edict.checkpoint.cpkRoot%comment
    fileName = 'checkpoint_%03d.pth.tar' % (epoch)
    os.makedirs(cpkRoot, exist_ok=True)
    filePath = os.path.join(cpkRoot, fileName)
    bestPath = os.path.join(cpkRoot, 'model_best.pth.tar')
    if isBest:
        shutil.copyfile(filePath, bestPath)
        return
    torch.save(state, filePath)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()
    args.lr = 0.2
    args.nEpochs = 40
    args.power = 0.9
    model = AlexNet(num_classes=2)
    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
    plt.figure()
    x = list(range(10))
    y = []

    for epoch in range(10):
        # scheduler.step(epoch)
        # y.append(scheduler.get_lr()[0])
        adjust_learning_rate(optimizer, epoch, args)
        y.append(optimizer.param_groups[0]['lr'])

    plt.plot(x, y)
    plt.show()
