import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
# import torchfields
from scipy.io import savemat

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    # 对应坐标与第几个，
    coord = make_coord(img.shape[-2:])
    # img打平， 形状为 [B, HxW， C]
    rgb = img.view(img.size(0), img.size(1), -1).permute(0, 2, 1)
    return coord, rgb

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

def get_kernel(kernlen=5, nsig=3):     # nsig 标准差 ，kernlen=16核尺寸
    nsig = nsig.detach().numpy()
    interval = (2*nsig+1.)/kernlen      #计算间隔
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)   #在前两者之间均匀产生数据

                                          #高斯函数其实就是正态分布的密度函数
    kern1d = np.diff(st.norm.cdf(x))      #先积分在求导是为啥？得到一个维度上的高斯函数值
    '''st.norm.cdf(x):计算正态分布累计分布函数指定点的函数值
        累计分布函数：概率分布函数的积分'''
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))   #np.outer计算外积，再开平方，从1维高斯参数到2维高斯参数
    kernel = kernel_raw/kernel_raw.sum()             #确保均值为1
    return kernel


class R_PSF_down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nsig = nn.Parameter(torch.FloatTensor(torch.rand([1])), requires_grad=True)
        kernel = get_kernel(3, torch.max(self.nsig, 0)[0])
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(c,c,1,1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, input):
        out = torch.nn.functional.conv2d(input, self.weight, stride=2, padding=1)
        return out

class Register(nn.Module):
    def __init__(self, hs_channels, patch_size):
        super(Register, self).__init__()
        # 注意力层
        self.q_linear = nn.Linear(1 * patch_size * patch_size,
                                  hs_channels * patch_size * patch_size)
        self.k_linear = nn.Linear(hs_channels * patch_size * patch_size,
                                  hs_channels * patch_size * patch_size)
        self.v_linear = nn.Linear(hs_channels * patch_size * patch_size,
                                  hs_channels * patch_size * patch_size)
        self.patchsize = patch_size

    def forward(self, hs, ms):  # input tensor size is [batch_size, seq_length, input_dim]
        """填充，让hs能被patch整切"""
        b, c, h_o, w_o = hs.shape
        pad_h = (4 - (h_o % 4)) % 4
        if pad_h:
            hs= torch.nn.functional.pad(hs, (0, pad_h, 0, pad_h))
        b, c, h, w = hs.shape
        'spilt cube to patch'
        hs_copy = hs.view(b, c, h // self.patchsize, self.patchsize, w // self.patchsize, self.patchsize)
        hs_copy = hs_copy.permute(0, 1, 2, 4, 3, 5)
        hs_copy = hs_copy.contiguous().view(b, c, -1, self.patchsize, self.patchsize)  # 使用.contiguous()来确保张量在内存中是连续的
        hs_copy = hs_copy.permute(0, 2, 1, 3, 4)
        hs_copy = hs_copy.flatten(2)

        """插值，输入hs做参考的时，取4个波段"""
        ms = torch.mean(ms, dim=1).unsqueeze(1)
        ms = torch.nn.functional.interpolate(ms, size=[h_o, w_o], mode='bicubic', align_corners=False)
        if pad_h:
            ms = torch.nn.functional.pad(ms, (0, pad_h, 0, pad_h))
        b, c_m, h, w = ms.shape

        ms_copy = ms.view(b, c_m, h // self.patchsize, self.patchsize, w // self.patchsize, self.patchsize)
        ms_copy = ms_copy.permute(0, 1, 2, 4, 3, 5)
        ms_copy = ms_copy.contiguous().view(b, c_m, -1, self.patchsize, self.patchsize)  # 使用.contiguous()来确保张量在内存中是连续的
        ms_copy = ms_copy.permute(0, 2, 1, 3, 4)
        ms_copy = ms_copy.flatten(2)

        ms_q = self.q_linear(ms_copy)
        hs_k = self.k_linear(hs_copy)
        hs_v = self.v_linear(hs_copy)
        scores = torch.matmul(ms_q, hs_k.transpose(1, 2))
        weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(weights, hs_v)
        out = out.view(b, out.size(1), c, self.patchsize, self.patchsize)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        if pad_h:
            out = out[:, :, :-pad_h, :-pad_h]
        return out


class B_T_Block(nn.Module):
    def __init__(self, in_c, args):
        super().__init__()
        self.args = args
        self.mlp = MLP(args.bands*9+4, in_c, [256, 256, 256, 256])
        self.register_block = Register(in_c, 4)
        self.gen_feat = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=256, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=in_c, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
        )
    def query(self, feat, coord, cell=None):

        # unflod成 [B,3*3,C,H,W]
        feat = torch.nn.functional.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        # feat = torch.nn.functional.unfold(feat, 1, padding=0).view(
        #     feat.shape[0], feat.shape[1] * 1, feat.shape[2], feat.shape[3])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda(self.args.gpu) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = torch.nn.functional.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = torch.nn.functional.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, x, h, ref):
        x = self.gen_feat(x)

        x_coord, _ = to_pixel_samples(torch.zeros(x.size(0), x.size(1), h, h).contiguous())
        x_coord = x_coord.unsqueeze(0).repeat(x.size(0),1,1).cuda(self.args.gpu)
        cell = torch.ones_like(x_coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / h

        y = self.query(x, x_coord, cell)
        y = y.view(y.size(0), h, h, y.size(-1)).permute(0, 3, 1, 2)
        out = self.register_block(y,ref)
        return out


class B_M(nn.Module):
    def __init__(self, in_c, args):
        super().__init__()
        self.args = args
        self.mlp = MLP(40, in_c, [256, 256, 256, 256])

    def forward(self, feat, coord, cell=None):

        # unflod成 [B,3*3,C,H,W]
        feat = torch.nn.functional.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda(self.args.gpu) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = torch.nn.functional.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = torch.nn.functional.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret


class R_Block(nn.Module):
    def __init__(self, in_c, args):
        super().__init__()
        self.args = args
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 4, kernel_size=1, stride=1),
        )
        self.B_M = B_M(4, args)
        self.gen_feat = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
        )
    def forward(self, x, h):
        y = self.conv(x)
        y = self.gen_feat(y)
        coord, _ = to_pixel_samples(torch.zeros(y.size(0), y.size(1), h, h).contiguous())
        coord = coord.unsqueeze(0).repeat(y.size(0),1,1).cuda(self.args.gpu)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / h
        y = self.B_M(y, coord, cell)
        y = y.view(y.size(0), h, h, y.size(-1)).permute(0, 3, 1, 2)
        return y


class B_M_UP(nn.Module):
    def __init__(self, in_c, args):
        super().__init__()
        self.args = args
        self.mlp = MLP(922, in_c, [256, 256, 256, 256])

    def forward(self, feat, coord, cell=None):

        # unflod成 [B,3*3,C,H,W]
        feat = torch.nn.functional.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda(self.args.gpu) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = torch.nn.functional.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = torch.nn.functional.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.mlp(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

class R_T_Block(nn.Module):
    def __init__(self, out_c, args):
        super().__init__()
        self.args = args
        self.B_M = B_M_UP(out_c, args)
        self.conv = nn.Sequential(
            nn.Conv2d(4, out_c, kernel_size=1, stride=1),
        )
        self.gen_feat = nn.Sequential(
            nn.Conv2d(in_channels=out_c, out_channels=256, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=out_c, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
        )

    def forward(self, x, h):
        y = self.conv(x)
        y = self.gen_feat(y)
        coord, _ = to_pixel_samples(torch.zeros(y.size(0), y.size(1), h, h).contiguous())
        coord = coord.unsqueeze(0).repeat(y.size(0),1,1).cuda(self.args.gpu)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / h
        y = self.B_M(y, coord, cell)
        y = y.view(y.size(0), h, h, y.size(-1)).permute(0, 3, 1, 2)
        return y

class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in ):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride = 2, padding=3 // 2)

        self.act =  torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


    def forward(self, input):
        out1 = self.act(self.conv1(input))
        f_e = self.conv4(out1)
        down = self.act(self.conv5(f_e))
        return f_e, down

class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in = 64 ):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e

class Decoding_Block(torch.nn.Module):
    def __init__(self,c_in ):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(c_in, 256, kernel_size=3, stride=2,padding=3 // 2)

        self.act =  torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])
        Deconv = self.up(input)

        return Deconv
    def forward(self, input, map):

        up = self.up(input, output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]] )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))

        out3 = self.conv3(cat)

        return out3

class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])

        Deconv = self.up(input)

        return Deconv
    def forward(self, input,map):

        up = self.up(input,  output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]] )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class FusionM(nn.Module):
    def __init__(self, R, R_T, B, B_T, bands, args):
        super().__init__()
        self.R = R
        self.R_T = R_T
        self.B = B
        self.B_T = B_T
        self.args = args
        self.eta = 2*torch.nn.Parameter(torch.tensor(2.))


    def forward(self, PAN, HS, Xpk, h):
        B_Xpk = self.B(Xpk, 20, PAN)
        E_HS = self.B_T(B_Xpk - HS, h, HS)
        R_Xpk = self.R(Xpk, PAN.size(-1))
        E_MS = self.R_T(R_Xpk - PAN, h)

        Uk = Xpk + self.eta*E_HS + self.eta*E_MS
        return Uk, B_Xpk, E_HS

class UnetSpatial(torch.nn.Module):
    def __init__(self, cin):
        super(UnetSpatial, self).__init__()

        self.Encoding_block1 = Encoding_Block(128)
        self.Encoding_block2 = Encoding_Block(128)
        self.Encoding_block3 = Encoding_Block(128)
        self.Encoding_block4 = Encoding_Block(128)
        self.Encoding_block_end = Encoding_Block_End(128)

        self.Decoding_block1 = Decoding_Block(256)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()

        self.fe_conv1 = torch.nn.Conv2d(in_channels=cin, out_channels=128, kernel_size=3, padding=3 // 2)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        sz = x.shape
        # x = x.view(-1,1,sz[2],sz[3])
        x = self.fe_conv1(x)

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)

        return decode0

class FandD(nn.Module):
    def __init__(self, R, R_T, B_T, Denoise, bands, args):
        super().__init__()

        self.F1 = FusionM(R, R_T, B_T, B_T, bands, args)
        self.F2 = FusionM(R, R_T, B_T, B_T, bands, args)
        self.F3 = FusionM(R, R_T, B_T, B_T, bands, args)
        self.D1 = Denoise
        self.D2 = Denoise
        self.D3 = Denoise

    def forward(self, PAN, HS, h):
        X0 = torch.nn.functional.interpolate(HS, size=[h,h], mode='bicubic', align_corners=False)
        X1, B_Xp1, E_HS1 = self.F1(PAN, HS, X0, h)
        X1 = self.D1(X1)
        X2, B_Xp2, E_HS2 = self.F2(PAN, HS, X1, h)
        X2 = self.D2(X2)
        X3, B_Xp3, E_HS3 = self.F3(PAN, HS, X2, h)
        X3 = self.D3(X3)

        return X3, B_Xp1, B_Xp2, B_Xp3, E_HS1, E_HS2, E_HS3


"""
    主框架
"""
class Main(nn.Module):
    def __init__(self, bands, args):
        super().__init__()
        R_T = R_T_Block(bands, args)
        R = R_Block(bands, args)
        B_T = B_T_Block(bands, args)
        D = UnetSpatial(bands)
        self.R = R
        self.B_T = B_T

        self.FandD = FandD(R, R_T, B_T, D, bands, args)

    def forward(self, PAN, HS, h):

        HS_H, B_Xp1, B_Xp2, B_Xp3, E_HS1, E_HS2, E_HS3 = self.FandD(PAN, HS, h)

        return HS_H, B_Xp1, B_Xp2, B_Xp3, E_HS1, E_HS2, E_HS3

