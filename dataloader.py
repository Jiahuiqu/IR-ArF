import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.io import loadmat
import torch as t
import random
import os
from skimage import transform,data
device = t.device('cuda:1' if t.cuda.is_available() else 'cpu')



class MyDataset(Dataset):
    def __init__(self, root, mode, alpha):
        super(MyDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.alpha = alpha
        self.transform_1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
        ])
        if mode == 'test' and self.root.split('/')[-1] != 'XJ':
            self.lrHSroot = os.listdir(os.path.join(root, "test", "LRHS"+ alpha))
            self.lrHSroot.sort(key=lambda x: int(x.split(".")[0]))
            self.PANroot = os.listdir(os.path.join(root, "test", "hrMS"))
            self.PANroot.sort(key=lambda x: int(x.split(".")[0]))
            self.gtHSroot = os.listdir(os.path.join(root, "test", "gtHS"))
            self.gtHSroot.sort(key=lambda x: int(x.split(".")[0]))
        elif mode == 'test':
            self.lrHSroot = os.listdir(os.path.join(root, "test", "LRHS" + alpha))
            self.lrHSroot.sort(key=lambda x: int(x.split(".")[0]))
            self.PANroot = os.listdir(os.path.join(root, "test", "hrMS"))
            self.PANroot.sort(key=lambda x: int(x.split(".")[0]))

        if mode == 'train':
            self.lrHSroot = os.listdir(os.path.join(root, "train", "LRHS"+ alpha))
            self.lrHSroot.sort(key=lambda x: int(x.split(".")[0]))
            self.PANroot = os.listdir(os.path.join(root, "train", "hrMS"))
            self.PANroot.sort(key=lambda x: int(x.split(".")[0]))
            self.PANroot = self.PANroot + self.PANroot + self.PANroot
            self.gtHSroot = os.listdir(os.path.join(root, "train", "gtHS"))
            self.gtHSroot.sort(key=lambda x: int(x.split(".")[0]))
            self.gtHSroot = self.gtHSroot + self.gtHSroot + self.gtHSroot

    def __getitem__(self, item):
        if self.root.split('/')[-1] == 'pavia':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS"+ self.alpha, self.lrHSroot[item]))['LRHS'].reshape(-1, 20, 20)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 80, 80)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['gtHS'].reshape(-1, 160, 160)
        elif self.root.split('/')[-1] == 'cave' and self.mode == 'train':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 64, 64)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 256, 256)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['gtHS'].reshape(-1, 512,
                                                                                                            512)
        elif self.root.split('/')[-1] == 'cave' and self.mode == 'test':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 64, 64)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 256, 256)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['gtHS'].reshape(-1, 512,
                                                                                                            512)
        elif self.root.split('/')[-1] == 'harvard' and self.mode == 'train':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 20, 20)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 80, 80)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['gtHS'].reshape(-1, 160,
                                                                                                            160)
        elif self.root.split('/')[-1] == 'harvard' and self.mode == 'test':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 30, 30)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 120, 120)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['blockData'].reshape(-1, 240,
                                                                                                            240)
        elif self.root.split('/')[-1] == 'KAUST' :
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 16, 16)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 64, 64)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['lrHS_crop'].reshape(-1, 128,
                                                                                                            128)
        elif self.root.split('/')[-1] == 'XJ' and self.mode == 'train':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 40, 40)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 160, 160)
            gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['gtHS'].reshape(-1, 160,
                                                                                                            160)
        elif self.root.split('/')[-1] == 'XJ' and self.mode == 'test':
            LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS" + self.alpha, self.lrHSroot[item]))[
                'LRHS'].reshape(-1, 40, 40)
            PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 160, 160)
            gtHS = torch.zeros(137, 160, 160)
        return LRHS, PAN, gtHS

    def __len__(self):
        return self.lrHSroot.__len__()




if __name__ == "__main__":
    root = "../dataset/pavia"
    data = MyDataset(root, "train", '_alpha1')
    print(data.__len__())
    print(data.__getitem__(0))
