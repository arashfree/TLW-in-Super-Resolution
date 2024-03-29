import argparse
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy as np
from networks import device,up_scale_factor,scale_factor,RCAN,VDSR,SRCNN,EDSR,weights_init,Weight,WeightStochastic,HATNet,UHATNet
from tlw import TLWLoss,TLWLossStochastic
from utils import record,SRResults
import lpips
import time
from tqdm import tqdm
import random
from load_data import load_data_matlab_degradation,SRMatlabDataset,SRMatlabDataset2
from val import validation




def load_data(train_path,val_path,batch_size):
    dataset_train = SRMatlabDataset(train_path,
                                                 scale_factor,
                                                 False, 'cpu')
    dataset_val = load_data_matlab_degradation(val_path,
                                               scale_factor,
                                               True, device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,num_workers=4, worker_init_fn=_worker_init_fn_)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    return dataloader_train,dataloader_val

def train(models,
          wmodels,
        tlws,
        dataloader_train,
        dataloader_val,
        epochs,
        device='cuda'
        ):
    # ,num_workers=8, worker_init_fn=_worker_init_fn_)

    startepoch, maxepoch = epochs
    tlwloss1,tlwloss2 = tlws
    w_losses = {}
    mean_ws = {}
    results = {}
    best_lpips = {}
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) #bug
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) #bug
    for key in models.keys():
        models[key].train()
        results[key] = SRResults(loss_fn_vgg)
        best_lpips[key] = 1.0
        if wmodels[key] is not None:
            wmodels[key].train()
        w_losses[key] = record()
        mean_ws[key] = record()
    start_time = time.time()

    for epoch in range(startepoch, maxepoch):
        k = 0.6 * (1 - np.exp(-epoch / 100    )) + 0.3
        tlwloss1.WNetStochastic.setk(k)
        # tlwloss2.WNetStochastic.setk(k)



        pbar = tqdm(dataloader_train, position=0)

        for index,(Hrs,Lrs) in enumerate(pbar):
            Hrs = Hrs.to(device)
            Lrs = Lrs.to(device)

            ## weights update
            with torch.no_grad():
                Srs = models['tlw1'](Lrs)
            w_loss, mean_w = tlwloss1.updateWeightNet(Srs, Hrs)
            w_losses['tlw1'].append(w_loss)
            mean_ws['tlw1'].append(mean_w) #bug

            # WNET_tlw2
            # with torch.no_grad():
            #     Srs = models['tlw2'](Lrs)
            # w_loss, mean_w = tlwloss2.updateWeightNet(Srs, Hrs)
            # w_losses['tlw2'].append(w_loss)
            # mean_ws['tlw2'].append(mean_w)


            ###tlw1
            optims['tlw1'].zero_grad()
            Srs = models['tlw1'](Lrs)
            loss = tlwloss1.forward(Srs, Hrs)
            loss.backward()
            optims['tlw1'].step()
            results['tlw1'].update(Srs, Hrs, Lrs, loss)

            # ###tlw2
            # optims['tlw2'].zero_grad()
            # Srs = models['tlw2'](Lrs)
            # loss = tlwloss2.forward(Srs, Hrs)
            # loss.backward()
            # optims['tlw2'].step()
            # results['tlw2'].update(Srs, Hrs, Lrs, loss)


            # ####base mse
            # optims['mse'].zero_grad()
            # Srs_base = models['mse'](Lrs)
            # loss_base = nn.MSELoss()(Srs_base, Hrs)  # *torch.mean(w.detach())
            # loss_base.backward()
            # optims['mse'].step()
            # results['mse'].update(Srs_base, Hrs, Lrs, loss_base)


            ####base L1
            optims['l1'].zero_grad()
            Srs_base = models['l1'](Lrs)
            loss_base = nn.L1Loss()(Srs_base, Hrs)  # *torch.mean(w.detach())
            loss_base.backward()
            optims['l1'].step()
            results['l1'].update(Srs_base, Hrs, Lrs, loss_base)


            ####base uncertainty
            optims['uncertainty'].zero_grad()
            Srs_uncertainty,uncertainty = models['uncertainty'](Lrs)
            #loss_uncertainty = torch.mean(torch.abs(Srs_uncertainty - Hrs) / uncertainty + 2 * torch.log(uncertainty))
            uncertainty = uncertainty - torch.min(uncertainty)
            loss_uncertainty = torch.mean(torch.abs(Srs_uncertainty - Hrs) * uncertainty.detach())

            loss_uncertainty.backward()
            optims['uncertainty'].step()
            results['uncertainty'].update(Srs_uncertainty, Hrs, Lrs, loss_uncertainty)



            pbar.set_description(('%10s' * 1 + '%10.4g' * 4) % (
                f'{epoch}/{maxepoch - 1}', np.mean(results['tlw1'].psnrs.inloopdata),np.mean(results['l1'].psnrs.inloopdata),np.mean(results['uncertainty'].psnrs.inloopdata) ,np.mean(w_losses['tlw1'].inloopdata) ))
            pbar.update(1)


        print('----------- Epoch: ', epoch)

        for key in results.keys(): #bug
            results[key].endloop()
            if wmodels[key] is not None:
                w_losses[key].endloop()
                mean_ws[key].endloop()

            results[key].print(key +' Loss: ')

        # print('w_losses:', w_losses['tlw1'].top(), '   ', w_losses['tlw2'].top(),
        #       'mean_ws:', mean_ws['tlw1'].top(), '   ', mean_ws['tlw2'].top())

        print(f"-------------------- Epoch:{epoch}, time:{time.time() - start_time}")

        valid_result = {}
        for key in models.keys(): #bug
            valid_result[key]= validation(models[key], dataloader_val,(loss_fn_vgg, loss_fn_alex),device)

            print("val "+key+":", valid_result[key])

        for key in models.keys(): #bug
            torch.save(models[key].state_dict(), opt.modelpath + 'SRModel_'+key+'.pth')
            if wmodels[key] is not None:
                torch.save(wmodels[key].state_dict(), opt.modelpath + 'WNet_' + key + '.pth')

        for key in models.keys(): #bug
            if valid_result[key]['LPIPS-vgg'] < best_lpips[key]:
                best_lpips[key] = valid_result[key]['LPIPS-vgg']
                torch.save(models[key].state_dict(), opt.modelpath +'SRModel_'+key+'_best.pth')
                if wmodels[key] is not None:
                    torch.save(wmodels[key].state_dict(), opt.modelpath +'WNet_'+key+'_best.pth')




def _worker_init_fn_(_):
    # torch_seed = torch.initial_seed()
    # np_seed = torch_seed // 2**32 - 1
    # random.seed(torch_seed)
    # np.random.seed(np_seed)
    # print(np_seed)
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='HAT', help='model structure')
    parser.add_argument('--trainpath', type=str, default='SRDataset/DIV2K_train_HR/DIV2K_train_HR/'
                        , help='train dataset path')
    parser.add_argument('--valpath', type=str, default=r'SRDataset/SR_testing_datasets/Set5/',
                        help='validation dataset path')
    parser.add_argument('--startepoch', type=int, default=0)
    parser.add_argument('--endepoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--modelpath', type=str, default=r'Checkpoints/L1TLW_L1UNCERTAINTY_L1/HATx4/', help='models path')
    parser.add_argument('--load', action='store_true', default=True, help='load models')
    parser.add_argument('--best', action='store_true', help='best model or last')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt



if __name__ == '__main__':
    opt = parse_opt(True)

    SRdict = {'RCAN':RCAN,'EDSR':EDSR,'VDSR':VDSR,'HAT':HATNet}
    SuperResolutionModel = SRdict[opt.model]

    SRModel = SuperResolutionModel(3).to(opt.device)
    device = opt.device
    models = {}
    wmodels = {}
    optims = {}
    woptims = {}

    if not opt.load:
        SRModel.apply(weights_init)
        torch.save(SRModel.state_dict(),  opt.modelpath + 'SRModel_start.pth')
        
        models['tlw1'] = SuperResolutionModel(3).to(device)
        optims['tlw1'] = torch.optim.Adam(models['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        #models['tlw2'] = SuperResolutionModel(3).to(device)
        #optims['tlw2'] = torch.optim.Adam(models['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['l1'] = SuperResolutionModel(3).to(device)
        optims['l1'] = torch.optim.Adam(models['l1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        #models['mse'] = SuperResolutionModel(3).to(device)
        #optims['mse'] = torch.optim.Adam(models['mse'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['uncertainty'] = UHATNet(3).to(device)
        optims['uncertainty'] = torch.optim.Adam(models['uncertainty'].parameters(), lr=0.0001, betas=(0.5, 0.999))

        wmodels['tlw1'] = WeightStochastic(0.5).to(device)
        #wmodels['tlw2'] = WeightStochastic(0.5).to(device)
        wmodels['l1'] = None
        #wmodels['mse'] = None
        wmodels['uncertainty'] = None

        # WNet_tlw1 = Weight(0.5).cuda()
        # WNet_tlw2 = Weight(0.5).cuda()

        woptims['tlw1'] = torch.optim.Adam(wmodels['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        #woptims['tlw2'] = torch.optim.Adam(wmodels['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))

        models['tlw1'].load_state_dict(SRModel.state_dict())
        #models['tlw2'].load_state_dict(SRModel.state_dict())
        models['l1'].load_state_dict(SRModel.state_dict())
        #models['mse'].load_state_dict(SRModel.state_dict())
        #wmodels['tlw1'].apply(weights_init)
        #wmodels['tlw2'].load_state_dict(wmodels['tlw1'].state_dict())

    else:

        models['tlw1'] = SuperResolutionModel(3).to(device)
        optims['tlw1'] = torch.optim.Adam(models['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        #models['tlw2'] = SuperResolutionModel(3).to(device)
        #optims['tlw2'] = torch.optim.Adam(models['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['l1'] = SuperResolutionModel(3).to(device)
        optims['l1'] = torch.optim.Adam(models['l1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        #models['mse'] = SuperResolutionModel(3).to(device)
        #optims['mse'] = torch.optim.Adam(models['mse'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        models['uncertainty'] = UHATNet(3).to(device)
        optims['uncertainty'] = torch.optim.Adam(models['uncertainty'].parameters(), lr=0.0001, betas=(0.5, 0.999))

        wmodels['tlw1'] = WeightStochastic(0.5).to(device)
        #wmodels['tlw2'] = WeightStochastic(0.5).to(device)
        woptims['tlw1'] = torch.optim.Adam(wmodels['tlw1'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        #woptims['tlw2'] = torch.optim.Adam(wmodels['tlw2'].parameters(), lr=0.0001, betas=(0.5, 0.999))
        wmodels['l1'] = None
        #wmodels['mse'] = None
        wmodels['uncertainty'] = None

        if opt.best:
            for key in  models.keys():
                models[key].load_state_dict(torch.load(opt.modelpath + 'SRModel_'+ key + '_best.pth', map_location=torch.device(device)))
                if wmodels[key] is not None:
                    wmodels[key].load_state_dict(
                        torch.load(opt.modelpath + 'WNet_' + key + '_best.pth', map_location=torch.device(device)))

        else:
            for key in models.keys():
                models[key].load_state_dict(torch.load(opt.modelpath + 'SRModel_' + key + '.pth', map_location=torch.device(device)))
                if wmodels[key] is not None:
                    wmodels[key].load_state_dict(
                        torch.load(opt.modelpath + 'WNet_' + key + '.pth', map_location=torch.device(device)))


    loss_fn_vgg = lpips.LPIPS(net='vgg',lpips=False).to(device)

    tlwloss1 = TLWLossStochastic(wmodels['tlw1'], loss_fn_vgg, woptims['tlw1'], 'l1',device=device)
    tlwloss2 = None#TLWLossStochastic(wmodels['tlw2'], loss_fn_vgg, woptims['tlw2'], 'l2',device=device)

    train_loader , val_loader = load_data(train_path= opt.trainpath,
          val_path= opt.valpath,
          batch_size=opt.batchsize)

    train(models,wmodels,(tlwloss1,tlwloss2),train_loader,val_loader,
          epochs=(opt.startepoch,opt.endepoch),device=device)

