


import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from networks import up_scale_factor
from torchvision.utils import make_grid

def calPSNR(Srs, Hrs, Lrs):
    up = nn.Upsample(scale_factor=up_scale_factor, align_corners=False, mode='bicubic')
    mseloss = nn.MSELoss()(Srs, Hrs)
    mseloss_bicubic = nn.MSELoss()(up(Lrs), Hrs)
    PSNR = -10 * np.log10(mseloss.item()) + 20 * np.log10(2)
    PSNR_bicubic = -10 * torch.log10(mseloss_bicubic).item() + 20 * np.log10(2)
    PSNR_def = PSNR - PSNR_bicubic
    return PSNR_def, PSNR, PSNR_bicubic



class record():
    def __init__(self, name='no name'):
        self.name = name
        self.inloopdata = []
        self.loopdata = []
        self.inloopsize = 0
        self.loopsize = 0

    def append(self, value):
        self.inloopdata.append(value)
        self.inloopsize += 1

    def endloop(self):
        self.loopdata += [sum(self.inloopdata) / self.inloopsize]
        self.loopsize += 1
        self.inloopdata = []
        self.inloopsize = 0

    def top(self):
        return self.loopdata[-1]

    def avreageall(self):
        return sum(self.loopdata) / self.loopsize
    def mean(self):
        return np.mean(self.loopdata)


    def loadfromnpy(self, filename):
        self.loopdata = np.load(filename).tolist()
        self.loopsize = len(self.loopdata)


class SRResults():
    def __init__(self, D):
        self.D = D
        self.lpips_losses = record()
        self.srlosses = record()
        self.L1losses = record()
        self.MSElosses = record()
        self.psnrs = record()

    def update(self, Srs, Hrs, Lrs, srloss):
        b = Srs.size(0)
        # print('Srs',torch.sum(Srs))
        for i in range(b):
            mseloss = torch.mean((Srs[i, :, :, :] - Hrs[i, :, :, :]) ** 2)
            PSNR = -10 * np.log10(mseloss.item() + 0.00001) + 20 * np.log10(2)
            self.MSElosses.append(mseloss.item())
            self.psnrs.append(PSNR.item())
        nloss = nn.L1Loss()(Srs, Hrs)
        self.srlosses.append(srloss.item())
        self.L1losses.append(nloss.item())
        self.lpips_losses.append(torch.mean(self.D(Srs, Hrs, normalize=False)).item())

    def endloop(self):
        self.lpips_losses.endloop()
        self.srlosses.endloop()
        self.L1losses.endloop()
        self.psnrs.endloop()
        self.MSElosses.endloop()

    def print(self, label):
        print(
            f"{label}\t---- PSNRS:{self.psnrs.top():0.8f}  L1 LOSS STANDARD: {self.L1losses.top():0.8f} MSE LOSS STANDARD: {self.MSElosses.top():0.4f}   lpips:{self.lpips_losses.top():0.4f}  SRLoss:{self.srlosses.top():0.4f}")

def cudausage():
    t = torch.cuda.get_device_properties('cuda').total_memory
    r = torch.cuda.memory_reserved('cuda')
    a = torch.cuda.memory_allocated('cuda')
    f = r-a  # free inside reserved
    print('total:',r/10**9,'\t usage:',a/10**9,'\t free:',f/10**9)

def show_tensor_images(image_tensor, filename='', num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor_c = image_tensor.clone()
    image_tensor[: ,0, :, :] = image_tensor_c[:, 2,:,:]
    image_tensor[:, 2, :, :] = image_tensor_c[:, 0, :, :]

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if (filename != ''):
        plt.imsave(filename, image_grid.permute(1, 2, 0).squeeze().numpy())
    plt.show()


