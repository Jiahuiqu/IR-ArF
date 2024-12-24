import torch
import torch.nn as nn
import random
from scipy.io import savemat
from torch import optim
import numpy as np
import argparse
from torch.utils.data import DataLoader
from model import Main
from dataloader import MyDataset
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import warnings
import shutil
import math
import time
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='pavia', choices=['pavia', 'houston'])
parser.add_argument('--data', default="../dataset/pavia", metavar='DIR')
parser.add_argument('--type', default='', choices=['', '_Elastic300', '_Elastic500', '_Homography2', '_Turbulance_4'])
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--bands', default=102, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='False', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default="0", type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.35, type=float, help='supervised contrastive loss weight')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--store_name_define', default='Dong_reg_addfeat', help='保存名字的尾巴')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(torch.nn.functional, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs),
         'lr', str(args.lr), args.store_name_define])
    if not os.path.exists(os.path.join('.', args.root_log, args.store_name)):
        os.mkdir(os.path.join('.', args.root_log, args.store_name))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    shutil.copyfile('./model.py', os.path.join('.', args.root_log, args.store_name, 'model.py'))

    argsDict = args.__dict__
    with open(os.path.join('.', args.root_log, args.store_name, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model")
    if args.dataset == 'pavia':
        model = Main(bands=102, args=args)
    elif args.dataset == 'cave':
        model = Main(bands=31, args=args)
    else:
        raise NotImplementedError('This dataset is not supported')
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """ optionally resume from a checkpoint """
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    """ end """

    """ dataset """
    train_dataset = MyDataset(args.data, "train", args.type)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset = MyDataset(args.data, "test", args.type)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    """ end """

    """ tensorboard """
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    """ end """

    """ loss """
    loss_l1 = nn.L1Loss()
    loss_NCC = NCC()
    """ end """

    best_loss = 200
    best_l1_loss = 200

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_lr(optimizer, epoch, args)

        # train for one epoch
        loss, l1_loss = train(train_loader, model, loss_l1, loss_NCC, optimizer, epoch, args, tf_writer)
        scheduler.step()

        # evaluate on validation set
        validate(val_loader, model, loss_l1, epoch, args, tf_writer)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if is_best:
            best_l1_loss = l1_loss
        print('Best loss: {:.3f}, Best L1loss: {:.3f}'.format(best_loss, best_l1_loss))

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, loss_l1, loss_NCC, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_all = AverageMeter('Loss', ':.4e')
    l1_loss_all = AverageMeter('L1_Loss', ':.4e')
    NCC_loss_all = AverageMeter('NCC_Loss', ':.4e')


    model.train()
    end = time.time()
    for step, (lrhs, pan, gths) in enumerate(train_loader):
        h = random.randint(80, 160)
        pan = pan.type(torch.float).cuda(args.gpu)
        lrhs = lrhs.type(torch.float).cuda(args.gpu)
        gths = gths.type(torch.float).cuda(args.gpu)
        gths_loss = torch.nn.functional.interpolate(gths, size=[h,h], mode='bilinear', align_corners=False)
        batch_size = pan.shape[0]

        HRHS, B_Xp1, B_Xp2, B_Xp3, E_HS1, E_HS2, E_HS3 = model(pan, lrhs, h)
        """LOSS区域"""
        loss1 = loss_l1(HRHS, gths_loss)
        #　配准ｌｏｓｓ
        loss2_1 = loss_NCC.loss(torch.mean(lrhs, dim=1).unsqueeze(1), torch.mean(B_Xp1, dim=1).unsqueeze(1))
        loss2_2 = loss_NCC.loss(torch.mean(lrhs, dim=1).unsqueeze(1), torch.mean(B_Xp2, dim=1).unsqueeze(1))
        loss2_3 = loss_NCC.loss(torch.mean(lrhs, dim=1).unsqueeze(1), torch.mean(B_Xp3, dim=1).unsqueeze(1))
        loss2_4 = loss_NCC.loss(torch.mean(gths_loss, dim=1).unsqueeze(1), torch.mean(E_HS1, dim=1).unsqueeze(1))
        loss2_5 = loss_NCC.loss(torch.mean(gths_loss, dim=1).unsqueeze(1), torch.mean(E_HS2, dim=1).unsqueeze(1))
        loss2_6 = loss_NCC.loss(torch.mean(gths_loss, dim=1).unsqueeze(1), torch.mean(E_HS3, dim=1).unsqueeze(1))
        loss2 = (loss2_1+loss2_2+loss2_3+loss2_4+loss2_5+loss2_6)/6

        loss = loss1+0.1*loss2

        loss_all.update(loss.item(), batch_size)
        l1_loss_all.update(loss1.item(), batch_size)
        NCC_loss_all.update(loss2.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss1 {l1_loss.val:.4f} ({l1_loss.avg:.4f})\t'
                      'LossNCC {NCC_loss.val:.4f} ({NCC_loss.avg:.4f})'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                loss=loss_all, l1_loss=l1_loss_all, NCC_loss = NCC_loss_all))
            print(output)

    tf_writer.add_scalar('Loss/train', loss_all.avg, epoch)
    tf_writer.add_scalar('L1 loss/train', l1_loss_all.avg, epoch)
    tf_writer.add_scalar('LR', l1_loss_all.avg, epoch)

    return loss_all.avg, l1_loss_all.avg

def validate(val_loader, model, loss_l1, epoch, args, tf_writer):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    l1_loss_all = AverageMeter('L1_Loss', ':.4e')

    with torch.no_grad():
        end = time.time()
        for step, (lrhs, pan, gths) in enumerate(val_loader):
            h = random.randint(80, 160)
            pan = pan.type(torch.float).cuda(args.gpu)
            lrhs = lrhs.type(torch.float).cuda(args.gpu)
            gths = gths.type(torch.float).cuda(args.gpu)
            gths_loss = torch.nn.functional.interpolate(gths, size=[h, h], mode='bilinear', align_corners=False)
            batch_size = pan.shape[0]
            HRHS, B_Xp1, B_Xp2, B_Xp3, E_HS1, E_HS2, E_HS3 = model(pan, lrhs, h)

            loss1 = loss_l1(HRHS, gths_loss)

            l1_loss_all.update(loss1.item(), batch_size)

            batch_time.update(time.time() - end)

        output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'l1_loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})'.format(
            epoch, len(val_loader), batch_time=batch_time, l1_loss=l1_loss_all, ))
        print(output)

        tf_writer.add_scalar('L1 loss/val', l1_loss_all.avg, epoch)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'MDF_NREG.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if state['epoch']%10==0:
        filename = os.path.join(args.root_log, args.store_name, 'MDF_NREG_'+str(state['epoch'])+'.pth.tar')
        torch.save(state, filename)


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()