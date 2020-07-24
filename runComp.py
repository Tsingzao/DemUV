from model import *
from dataset import *
from lib.config import options


if __name__ == '__main__':
    args = options()

    trainTransforms = Compose([
        RandomCrop(169),
        Normalize(),
        ToTensor()
    ])
    validTransforms = Compose([
        Normalize(),
        ToTensor()
    ])
    trainLoader = DataLoader(LAPSData(transforms=trainTransforms), batch_size=args.batchSize, shuffle=True)
    validLoader = DataLoader(LAPSData(transforms=validTransforms, timeRange='2017110100,2018010100'), batch_size=1, shuffle=False)

    # model = SubPixelTrainer(args, trainLoader, validLoader)
    # model = SRCNNTrainer(args, trainLoader, validLoader)
    # model = FSRCNNTrainer(args, trainLoader, validLoader)
    # model = DRCNTrainer(args, trainLoader, validLoader)
    model = EDSRTrainer(args, trainLoader, validLoader)
    model.run()

