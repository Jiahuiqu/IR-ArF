import torch
import torch as t
import torch.nn as nn
from scipy.io import savemat
from torch.utils.data import DataLoader
from model import Main
from dataloader import MyDataset
import argparse
from matplotlib import pyplot as plt
import time
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

# 超参数

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default="1", type=int,
                    help='GPU id to use.')
parser.add_argument('--bands', default=102, type=int)

def test(root, type):
    args = parser.parse_args()
    # 数据准备
    data = MyDataset(root, "test", type)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    # 模型加载
    model = Main(bands=102, args=args)
    model = model.cuda(args.gpu)
    model.load_state_dict(
        t.load('./MDF_NREG.best.pth.tar')[
            'state_dict'])
    model.eval()
    loss_fun = nn.L1Loss()
    h = 80
    # 模型测试
    start = time.time()
    with torch.no_grad():
        for step, (lrhs, pan, gths) in enumerate(data_loader):
            pan = pan.type(t.float).cuda(args.gpu)
            lrhs = lrhs.type(t.float).cuda(args.gpu)
            gths = gths.type(t.float).cuda(args.gpu)
            gths_loss = t.nn.functional.interpolate(gths, size=[h, h], mode='bilinear', align_corners=False)

            HRHS, _,_,_,_,_,_ = model(pan, lrhs, h)
            loss = loss_fun(HRHS, gths_loss)
            #
            print('step:' + str(step), '--loss:' + str(loss.data))
            savemat('./dataout/pavia_' + str(h) + '/'+str(step+1)+".mat", {'out': HRHS.cpu().detach().numpy()})
            savemat('./dataout/pavia_gtHS/'+str(step+1)+".mat", {'gtHS': gths_loss.cpu().detach().numpy()})
    end = time.time()
    print(end-start)
if __name__ == "__main__":
    root = "../dataset/pavia"

    test(root, '_Homography3')
