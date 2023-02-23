import argparse
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy as np
from networks import device,up_scale_factor,scale_factor,RCAN,VDSR,SRCNN,EDSR,weights_init,Weight,WeightStochastic
import lpips
import random
from load_data import load_data_matlab_degradation


from tqdm import tqdm
from utils import record
import cv2
import torch


def validation(model, dataloader , lpips_models,device = 'cuda'):
    loss_fn_vgg, loss_fn_alex = lpips_models

    l1 = record()
    mse = record()
    psnrAll = record()
    pi = record()
    pialex = record()
    model.eval()
    
    len_data = len(dataloader)

    with tqdm(total=len_data) as pbar:
        for index, (Hr, Lr) in enumerate(dataloader):
            Hr = Hr.to(device)
            Lr = Lr.to(device)

            with torch.no_grad():
                Sr = model(Lr).clamp(min=-1, max=1)


            pi.append(torch.mean(loss_fn_vgg(Sr, Hr, normalize=False)).item())

            pialex.append(torch.mean(loss_fn_alex(Sr, Hr, normalize=False)).item())


            l1.append(torch.mean(torch.abs(Sr - Hr)).item())

            mse.append(torch.mean((Sr - Hr) ** 2).item())

            Sr = ((Sr + 1 ) /2.0  )  # **2.2
            Hr = ((Hr + 1 ) /2.0  )  # **2.2

            sr_image = torch.permute(Sr, (0, 2, 3, 1)).detach().squeeze().cpu().numpy().astype('float32')
            hr_image = torch.permute(Hr, (0, 2, 3, 1)).detach().squeeze().cpu().numpy().astype('float32')


            sr_ycbcr_image = cv2.cvtColor(sr_image.astype('float32'),
                                          cv2.COLOR_BGR2YCrCb)  # bgr2ycbcr(lr_image, use_y_channel=False)
            hr_ycbcr_image = cv2.cvtColor(hr_image.astype('float32'),
                                          cv2.COLOR_BGR2YCrCb)  # imgproc.bgr2ycbcr(hr_image, use_y_channel=False)


            sr_y_image, sr_cb_image, sr_cr_image = cv2.split(sr_ycbcr_image)
            hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

            sr_y_tensor = torch.tensor(sr_y_image).to(device).unsqueeze_(0)
            hr_y_tensor = torch.tensor(hr_y_image).to(device).unsqueeze_(0)

            sr_y_tensor = 16 + 219 * sr_y_tensor
            hr_y_tensor = 16 + 219 * hr_y_tensor

            max_ = torch.tensor(255)
            total_psnr = 20 * torch.log10(max_) - 10. * torch.log10(torch.mean((sr_y_tensor - hr_y_tensor) ** 2))


            psnrAll.append(total_psnr.item())


            l1.endloop()

            mse.endloop()

            psnrAll.endloop()

            pi.endloop()

            pialex.endloop()

            pbar.set_description(('%10s' * 1) % (
                f'{index}/{len_data}'))
            pbar.update(1)

    return {"L1": l1.avreageall(), "MSE": mse.avreageall(), "PSNR": psnrAll.mean(),
            "LPIPS-vgg": pi.avreageall(),
            "LPIPS-alex": pialex.avreageall()}


def val(models,
        lpips_models,
        folder=r'E:\arash_file\sharif UNI 1399\IPL\Trainable Loss Weights\jetco_pc\superresolution\SRdataset-valid\dataset\SR_testing_datasets/',
        name_datasets=['Urban100', 'Set5', 'Set14', 'BSDS100',
                       'Manga109', 'T91', 'BSDS200', 'General100']
        , devie='cuda'):
    for key in models.keys():
        models[key].eval()
    up = nn.Upsample(scale_factor=up_scale_factor, align_corners=False, mode='bicubic')

    for name in name_datasets:
        dataset_val = load_data_matlab_degradation(
            folder + '/' + name,
            scale_factor, True, device)

        dataloader_val = DataLoader(dataset_val,
                                    batch_size=1,
                                    shuffle=False)  # ,num_workers=8, worker_init_fn=_worker_init_fn_)

        print('-----------', name, '-----------------')
        valid_result = {}
        for key in models.keys():
            valid_result[key] = validation(models[key], dataloader_val, (loss_fn_vgg, loss_fn_alex), 'cpu')
            print(print("\nval " + key + " :", valid_result[key]))
        val_bic = validation(up, dataloader_val, lpips_models, devie)
        print("\nval bicubic: ", val_bic)


def _worker_init_fn_(_):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EDSR', help='model structure')
    parser.add_argument('--folder', type=str, default='', help='models path')
    parser.add_argument('--modelpath',type=str,default='')
    parser.add_argument('--load', action='store_true', default=True, help='load models')
    parser.add_argument('--best', action='store_true', help='best model or last')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt(True)
    SRdict = {'RCAN':RCAN,'EDSR':EDSR,'VDSR':VDSR}
    SuperResolutionModel = SRdict[opt.model]

    device = opt.device

    SRModel = SuperResolutionModel(3).to(opt.device)
    models = {}
    wmodels = {}
    optims = {}
    woptims = {}

    if not opt.load:
        SRModel.apply(weights_init)
        torch.save(SRModel.state_dict(), './SRModel_start.pth')
        # SRModel.load_state_dict(torch.load('./SRModel_start.pth'))

        models['tlw1'] = SuperResolutionModel(3).to(device)
        optims['tlw1'] = torch.optim.Adam(models['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['tlw2'] = SuperResolutionModel(3).to(device)
        optims['tlw2'] = torch.optim.Adam(models['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['l1'] = SuperResolutionModel(3).to(device)
        optims['l1'] = torch.optim.Adam(models['l1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['mse'] = SuperResolutionModel(3).to(device)
        optims['mse'] = torch.optim.Adam(models['mse'].parameters(), lr=0.0001, betas=(0.5, 0.999))

        wmodels['tlw1'] = WeightStochastic(0.5).to(device)
        wmodels['tlw2'] = WeightStochastic(0.5).to(device)
        wmodels['l1'] = None
        wmodels['mse'] = None

        # WNet_tlw1 = Weight(0.5).cuda()
        # WNet_tlw2 = Weight(0.5).cuda()

        woptims['tlw1'] = torch.optim.Adam(wmodels['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        woptims['tlw2'] = torch.optim.Adam(wmodels['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))

        models['tlw1'].load_state_dict(SRModel.state_dict())
        models['tlw2'].load_state_dict(SRModel.state_dict())
        models['l1'].load_state_dict(SRModel.state_dict())
        models['mse'].load_state_dict(SRModel.state_dict())
        wmodels['tlw1'].apply(weights_init)
        wmodels['tlw2'].load_state_dict(wmodels['tlw1'].state_dict())

    else:

        models['tlw1'] = SuperResolutionModel(3).to(device)
        optims['tlw1'] = torch.optim.Adam(models['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['tlw2'] = SuperResolutionModel(3).to(device)
        optims['tlw2'] = torch.optim.Adam(models['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['l1'] = SuperResolutionModel(3).to(device)
        optims['l1'] = torch.optim.Adam(models['l1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['mse'] = SuperResolutionModel(3).to(device)
        optims['mse'] = torch.optim.Adam(models['mse'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        wmodels['tlw1'] = WeightStochastic(0.5).to(device)
        wmodels['tlw2'] = WeightStochastic(0.5).to(device)
        woptims['tlw1'] = torch.optim.Adam(wmodels['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        woptims['tlw2'] = torch.optim.Adam(wmodels['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        wmodels['l1'] = None
        wmodels['mse'] = None

        if opt.best:
            for key in models.keys():
                models[key].load_state_dict(
                    torch.load(opt.folder + 'SRModel_' + key + '_best.pth', map_location=torch.device(device)))
                if wmodels[key] is not None:
                    wmodels[key].load_state_dict(
                        torch.load(opt.modelpath + 'WNet_' + key + '_best.pth', map_location=torch.device(device)))

        else:
            for key in models.keys():
                models[key].load_state_dict(torch.load(opt.modelpath + 'SRModel_' + key + '.pth', map_location=torch.device(device)))


    name_datasets = ['Urban100', 'Set5', 'Set14', 'BSDS100',
                     'Manga109', 'T91', 'BSDS200', 'General100']

    loss_fn_alex = None#lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = None#lpips.LPIPS(net='vgg').to(device)
    val(models, (loss_fn_vgg, loss_fn_alex), opt.folder, name_datasets, opt.device)


