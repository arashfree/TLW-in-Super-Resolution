
import torch
from torch import nn
from resize_utils import imresize
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import imagesize


class SRMatlabDataset(Dataset):
    def __init__(self, path, scale_factor, source_size=False, device='cpu',transform=None):
        self.path = path
        self.file_names = os.listdir(path)
        self.image_mozaeic = 336
        self.image_hr_size = (336, 336)
        self.device = device
        self.source_size = source_size
        self.scale_factor = scale_factor

        self.len = 0
        self.flags = []
        self.ref_img = []
        self.start_load = []
        for i,filename in enumerate(self.file_names):
            filepath = os.path.join(self.path, filename)
            width, height = imagesize.get(filepath)
            width = width - width % (4 * 7)
            height = height - height % (4 * 7)
            length = (width // self.image_mozaeic) * (height // self.image_mozaeic)
            self.start_load.append(len(self.ref_img))
            for _ in range(length):
                self.ref_img.append(i)
            self.flags.append(False)

            self.len += length
        self.Dataset_lr = []
        self.Dataset_hr = []

        for _ in range(self.len):
            self.Dataset_lr.append(None)
            self.Dataset_hr.append(None)

        # self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        self.transform = transform
        # self.target_transform = target_transform
        self.q = 0
        self.p = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.flags[self.ref_img[idx]]:
            if self.Dataset_hr[idx] is None or self.Dataset_lr[idx] is None :
                print('IDX:',idx,self.ref_img[idx],self.flags[self.ref_img[idx]])
                print([(self.start_load[self.ref_img[idx]]+j,self.ref_img[self.start_load[self.ref_img[idx]]+j]) for j in range(30) ])

            self.p +=1
            return self.Dataset_hr[idx], self.Dataset_lr[idx]
        else:
            hr_image_path = os.path.join(self.path, self.file_names[self.ref_img[idx]])
            # print(idx, self.ref_img[idx], hr_image_path, self.start_load[self.ref_img[idx]],
            #       self.flags[self.ref_img[idx]])
            self.flags[self.ref_img[idx]] = True
            self.q +=1
            hr_image = cv2.imread(hr_image_path).astype(np.float64) / 255.0
            hr_image_height, hr_image_width = hr_image.shape[:2]
            hr_image_height_remainder = hr_image_height % (4 * 7)  ###change
            hr_image_width_remainder = hr_image_width % (4 * 7)  ###change
            hr_image = hr_image[:hr_image_height - hr_image_height_remainder,
                       :hr_image_width - hr_image_width_remainder, ...]

            h, w, _ = hr_image.shape

            if self.source_size:# or h * w < self.image_hr_size[0] * self.image_hr_size[1]:

                lr_image = imresize(hr_image, self.scale_factor, method='bicubic')

                hr_tensor = torch.permute(torch.tensor(hr_image).to(self.device), (2, 0, 1))
                lr_tensor = torch.permute(torch.tensor(lr_image).to(self.device), (2, 0, 1))

                self.Dataset_hr[self.start_load[self.ref_img[idx]]] = hr_tensor.float() * 2.0 - 1.0
                self.Dataset_lr[self.start_load[self.ref_img[idx]]] = lr_tensor.float() * 2.0 - 1.0
            else:
                data_index = 0
                for i in range(w // self.image_mozaeic):
                    for j in range(h // self.image_mozaeic):
                        w0 = i * self.image_mozaeic
                        w1 = (i * self.image_mozaeic) + self.image_hr_size[0]
                        h0 = j * self.image_mozaeic
                        h1 = (j * self.image_mozaeic) + self.image_hr_size[1]
                        if w1 <= w and h1 <= h:
                            hr_new = hr_image[h0:h1, w0:w1, :]  # .cuda()
                            lr_new = imresize(hr_new, self.scale_factor, method='bicubic')

                            hr_tensor = torch.permute(torch.tensor(hr_new).to(self.device), (2, 0, 1))
                            lr_tensor = torch.permute(torch.tensor(lr_new).to(self.device), (2, 0, 1))
                            #print(self.start_load[self.ref_img[idx]],data_index)
                            self.Dataset_hr[self.start_load[self.ref_img[idx]] + data_index] =  hr_tensor.float() * 2.0 - 1.0
                            self.Dataset_lr[self.start_load[self.ref_img[idx]] + data_index] = lr_tensor.float() * 2.0 - 1.0
                            data_index +=1
            # print(idx,self.ref_img[idx],hr_image_path,data_index,self.start_load[self.ref_img[idx]],self.flags[self.ref_img[idx]])

            return self.__getitem__(idx)



class SRMatlabDataset2(Dataset):
    def __init__(self, path, scale_factor, source_size=False, device='cpu',transform=None):
        self.path = path
        self.file_names = os.listdir(path)
        self.image_mozaeic = 336
        self.image_hr_size = (336, 336)
        self.device = device
        self.source_size = source_size
        self.scale_factor = scale_factor

        self.Dataset_lr = []
        self.Dataset_hr = []

        for i,filename in enumerate(self.file_names):
            hr_image_path = os.path.join(self.path, filename)
            print(hr_image_path)
            hr_image = cv2.imread(hr_image_path).astype(np.float64) / 255.0
            hr_image_height, hr_image_width = hr_image.shape[:2]
            hr_image_height_remainder = hr_image_height % (4 * 7)  ###change
            hr_image_width_remainder = hr_image_width % (4 * 7)  ###change
            hr_image = hr_image[:hr_image_height - hr_image_height_remainder,
                       :hr_image_width - hr_image_width_remainder, ...]

            h, w, _ = hr_image.shape

            if self.source_size:# or h * w < self.image_hr_size[0] * self.image_hr_size[1]:

                lr_image = imresize(hr_image, self.scale_factor, method='bicubic')

                hr_tensor = torch.permute(torch.tensor(hr_image).to(self.device), (2, 0, 1))
                lr_tensor = torch.permute(torch.tensor(lr_image).to(self.device), (2, 0, 1))

                self.Dataset_hr += [hr_tensor.float() * 2.0 - 1.0]
                self.Dataset_lr += [lr_tensor.float() * 2.0 - 1.0]
            else:
                for i in range(w // self.image_mozaeic):
                    for j in range(h // self.image_mozaeic):
                        w0 = i * self.image_mozaeic
                        w1 = (i * self.image_mozaeic) + self.image_hr_size[0]
                        h0 = j * self.image_mozaeic
                        h1 = (j * self.image_mozaeic) + self.image_hr_size[1]
                        if w1 <= w and h1 <= h:
                            hr_new = hr_image[h0:h1, w0:w1, :]  # .cuda()
                            lr_new = imresize(hr_new, self.scale_factor, method='bicubic')

                            hr_tensor = torch.permute(torch.tensor(hr_new).to(self.device), (2, 0, 1))
                            lr_tensor = torch.permute(torch.tensor(lr_new).to(self.device), (2, 0, 1))
                            #print(self.start_load[self.ref_img[idx]],data_index)
                            self.Dataset_hr += [hr_tensor.float() * 2.0 - 1.0]
                            self.Dataset_lr += [lr_tensor.float() * 2.0 - 1.0]

        self.transform = transform

    def __len__(self):
        return len(self.Dataset_hr)

    def __getitem__(self, idx):
        return self.Dataset_hr[idx], self.Dataset_lr[idx]

def load_data_matlab_degradation(path, scale_factor, source_size=False, device='cpu'):
    print('path',path)
    file_names = os.listdir(path)
    Dataset_hr = []
    Dataset_lr = []
    image_mozaeic = 512
    image_hr_size = (512, 512)

    total_files = len(file_names)


    for index in range(total_files):
        hr_image_path = os.path.join(path, file_names[index])
        hr_image = cv2.imread(hr_image_path).astype(np.float64) / 255.0
        hr_image_height, hr_image_width = hr_image.shape[:2]
        hr_image_height_remainder = hr_image_height % (4*7) ###change
        hr_image_width_remainder = hr_image_width % (4*7)   ###change
        hr_image = hr_image[:hr_image_height - hr_image_height_remainder,
                   :hr_image_width - hr_image_width_remainder, ...]

        h ,w ,_ = hr_image.shape

        if source_size or  h * w < image_hr_size[0] * image_hr_size[1] :

            lr_image = imresize(hr_image, scale_factor , method='bicubic')

            hr_tensor = torch.permute(torch.tensor(hr_image).to(device), ( 2, 0, 1))
            lr_tensor = torch.permute(torch.tensor(lr_image).to(device), ( 2, 0, 1))



            Dataset_hr += [hr_tensor.float() * 2.0 - 1.0]
            Dataset_lr += [lr_tensor.float() * 2.0 - 1.0]
        else:
            for i in range(w // image_mozaeic):
                for j in range(h // image_mozaeic):
                    w0 = i * image_mozaeic
                    w1 = (i * image_mozaeic) + image_hr_size[0]
                    h0 = j * image_mozaeic
                    h1 = (j * image_mozaeic) + image_hr_size[1]
                    if w1 <= w and h1 <= h:
                        hr_new = hr_image[h0:h1 ,w0:w1 ,:  ]  # .cuda()
                        lr_new = imresize(hr_new, scale_factor, method='bicubic')

                        hr_tensor = torch.permute(torch.tensor(hr_new).to(device), ( 2, 0, 1))
                        lr_tensor = torch.permute(torch.tensor(lr_new).to(device), ( 2, 0, 1))

                        Dataset_hr += [hr_tensor.float() * 2.0 - 1.0]
                        Dataset_lr += [lr_tensor.float() * 2.0 - 1.0]

    dataset = [(a, b) for a, b in zip(Dataset_hr, Dataset_lr)]

    return dataset
