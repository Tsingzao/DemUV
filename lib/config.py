from easydict import EasyDict as Edict
import argparse


edict = Edict()

edict.data = Edict()
edict.data.LAPSFormat = "/home/aiw/mntData/data168/data/SR_LAPS1H/%s/%s/MSP1_PMSC_AIWSRPF_LAPS-1H-0p01_L88_CZ_%s00_00000-00000.nc"
edict.data.LAPSNPY = './dataset/InputData/%s.npy'
edict.data.GRAPESMESOFormat = "/home/aiw/mntData/data47/data/AIWSLL/GRAPESMESOORI/%s/%s/MSP2_PMSC_AIWSR%sF_GRAPESMESOORI-0P03-1H_L88_CZ_%s00_00000-01200.nc"
edict.data.GRAPESMESOVariable = '%s-component_of_wind_height_above_ground'
edict.data.GRAPESMESONPY = './dataset/TestData/%s.npy'

edict.checkpoint = Edict()
edict.checkpoint.cpkRoot = './checkpoint_%s'



def options():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.001')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--upscale_factor', '-uf', type=int, default=3, help="super resolution upscale factor")
    parser.add_argument('--gpuID', type=str, default='3', help='GPU id for running')
    parser.add_argument('--power', type=float, default=0.9, help='lr power (default: 0.9)')
    parser.add_argument('--comment', type=str, default='EDSR', help='comment details')
    args = parser.parse_args()
    return args