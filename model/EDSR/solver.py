from lib.util import saveCheckpoint, adjust_learning_rate
import torch
import torch.backends.cudnn as cudnn
from model.EDSR.model import Net
from progress_bar import progress_bar
from tensorboardX import SummaryWriter


class EDSRTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(EDSRTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:%s' % config.gpuID if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.baseline = 999999
        self.config = config
        self.writer = None
        self.epoch = 0

    def build_model(self):
        self.model = Net(upscale_factor=self.upscale_factor).to(self.device)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100],
                                                              gamma=0.5)  # lr decay
        self.writer = SummaryWriter(comment=self.config.comment)

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data) in enumerate(self.training_loader):
            data, target = data[:,:2,::3,::3].to(self.device), data[:,:2].to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f, LR: %.4f' % (train_loss / (batch_num + 1), self.optimizer.param_groups[0]['lr']))
            self.writer.add_scalar('Train/TrainIterLoss_%s'%self.config.comment, loss.item(), self.epoch * len(self.training_loader) + batch_num)
            self.writer.add_scalar('Train/TrainAvgLoss_%s'%self.config.comment, train_loss / (batch_num + 1), self.epoch * len(self.training_loader) + batch_num)

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data) in enumerate(self.testing_loader):
                data, target = data[:,:2,::3,::3].to(self.device), data[:,:2].to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                avg_psnr += mse.item()
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
                self.writer.add_scalar('Valid/ValidIterLoss_%s'%self.config.comment, mse.item(), self.epoch * len(self.testing_loader) + batch_num)
                self.writer.add_scalar('Valid/ValidAvgLoss_%s'%self.config.comment, avg_psnr / (batch_num + 1), self.epoch * len(self.testing_loader) + batch_num)

        print("    Average Loss: {:.4f}".format(avg_psnr / len(self.testing_loader)))
        return avg_psnr / len(self.testing_loader)

    def run(self):
        self.build_model()
        for epoch in range(self.nEpochs):
            self.epoch = epoch
            adjust_learning_rate(self.optimizer, epoch, self.config)
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            saveCheckpoint(epoch, state, self.config.comment)
            validLoss = self.test()
            if validLoss < self.baseline:
                print('Current loss %.4f < Baseline %.4f ' % (validLoss, self.baseline))
                self.baseline = validLoss
                saveCheckpoint(epoch, state, self.config.comment, True)
            else:
                print('Current loss %.4f > Baseline %.4f ' % (validLoss, self.baseline))
            # self.scheduler.step(epoch)
