
import torch
from torch import nn
from resize_utils import imresize
import os
import cv2
import numpy as np



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
        hr_image_height_remainder = hr_image_height % 12
        hr_image_width_remainder = hr_image_width % 12
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
