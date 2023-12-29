import torch
import torch.nn as nn
from args import args

import kornia
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia
import skimage.measure
from util import cc
from multi_scale import Block1
from net import Decoder,Encoder,FusionDataset

# =============================================================================
# Hyperparameters Setting
# =============================================================================
#Train_data_choose = 'FLIR'  # 'FLIR' & 'NIR'
#if Train_data_choose == 'FLIR':
train_data_path = './Train_data_FLIR/'

root_VIS = './Train_data_FLIR/VIS/VIS'

root_IR = './Train_data_FLIR/IR/IR'
train_path = './Train_result'
device = "cuda"

batch_size =12
#channel = 64
epochs =160

lr = 1e-3

Train_Image_Number = len(os.listdir(train_data_path+'/IR/IR'))

Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
# =============================================================================
# Preprocessing and dataset establishmentwseff
# =============================================================================

# transforms = transforms.Compose([
#     transforms.Grayscale(1),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=10),
#     transforms.RandomResizedCrop(size=(224,224),scale=(0.8,1.2),ratio=(0.75,1.333)),
#     transforms.ToTensor(),
# ])
crop_size = (128, 128)
num_crops = 10
vis_image_paths = sorted([os.path.join(root_VIS, file) for file in os.listdir(root_VIS)])
ir_image_paths = sorted([os.path.join(root_IR, file) for file in os.listdir(root_IR)])
fusion_dataset = FusionDataset(vis_image_paths, ir_image_paths, crop_size, num_crops)
# Data_VIS = torchvision.datasets.ImageFolder(root_VIS, transform=transforms)
dataloader=torch.utils.data.DataLoader(fusion_dataset,batch_size=12,shuffle=True,num_workers=2)
# dataloader_VIS = torch.utils.data.DataLoader(Data_VIS, batch_size, shuffle=True)
#
# # Data_IR = torchvision.datasets.ImageFolder(root_IR, transform=transforms)
# dataloader_IR = torch.utils.data.DataLoader(Data_IR, batch_size, shuffle=True)

# =============================================================================
# Models
# =============================================================================

decoder=Decoder()

encoder=Encoder(16)

USE_MULTI_GPU = True


# 检测机器是否有多张显卡

if USE_MULTI_GPU and torch.cuda.device_count() > 1:

    MULTI_GPU = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    device_ids = [0, 1]

else:

    MULTI_GPU = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if MULTI_GPU:

    encoder= nn.DataParallel(encoder,device_ids=device_ids)
    decoder=nn.DataParallel(decoder,device_ids=device_ids)

decoder.to(device)
encoder.to(device)
#encoder= encoder.cuda()
#print('layer', next(layer.parameters()).device)
#decoder= decoder.cuda()
#print(next(net.parameters()).device)
#weight=torch.nn.Parameter(torch.ones(5),requires_grad=True)
#print(next(decoder.parameters()).device)
optimizer1 = optim.Adam(encoder.parameters(), lr=lr)
optimizer2 = optim.Adam(decoder.parameters(), lr=lr)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs//3, epochs//3*2], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2,  [epochs//3, epochs//3*2], gamma=0.1)
'''
if MULTI_GPU:
    optimizer1=nn.DataParallel(optimizer1, device_ids=device_ids)
    optimizer2=nn.DataParallel(optimizer2, device_ids=device_ids)
    scheduler1=nn.DataParallel(scheduler1, device_ids=device_ids)
    scheduler2=nn.DataParallel(scheduler2, device_ids=device_ids)
'''
MSELoss=nn.MSELoss()
#SmoothL1Loss = nn.SmoothL1Loss()
huber=nn.HuberLoss()
L1Loss = nn.L1Loss()
ssim = kornia.losses.SSIMLoss(3, reduction='mean')
MSE_LOSS=kornia.losses.MS_SSIMLoss()
# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
loss_train =[]
huber_loss_train =[]

cc_loss_train=[]

Gradient_loss_train =[]

ms_ssim_loss_train =[]
ssim_loss_train=[]
lr_list1 = []
lr_list2 = []
lr_list3=[]
alpha_list = []
for iteration in range(epochs):
    im_ir = 0
    im_vis = 0
    decoder.train()
    encoder.train()

    #block.train()
    for _,(data_vis,data_ir) in enumerate(dataloader):

        data_iter_VIS = iter(data_vis)
        data_iter_IR = iter(data_ir)
        for step in range(Iter_per_epoch):
            data_VIS= next(data_iter_VIS)
            data_IR = next(data_iter_IR)
            if MULTI_GPU:
                data_VIS = data_VIS.to(device)
                #print('DATA',data_VIS.device)
                data_IR = data_IR.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()


            # =====================================================================
            # Calculate loss
            # =====================================================================
           # data_IR = data_IR.cuda()
            f_ir=encoder(data_IR)
            f_v=encoder(data_VIS)
            #print("aaaaa",net.device)
            #net=net.to(torch.device('cuda'))
            #print("sss",next(net.trans1.conv.parameters()).device)
            #encoder
            F_IR=f_ir
            F_V=f_v

            img_recon_I=decoder(F_IR)
            img_recon_V=decoder(F_V)
            #mse_loss_VF = 5 * ssim(data_VIS, img_recon_V) + MSELoss(data_VIS, img_recon_V)
            #print("mse_loss_vf", mse_loss_VF)
            #mse_loss_IF = 5 * ssim(data_IR, img_recon_I) + MSELoss(data_IR, img_recon_I)
            #print("mse_loss_if", mse_loss_VF)
            # print(img_recon_V.size())

            huber_loss=MSELoss(data_VIS,img_recon_V)+MSELoss(data_IR,img_recon_I)
            Gradient_loss_V= L1Loss(
                kornia.filters.SpatialGradient()(data_VIS),
                kornia.filters.SpatialGradient()(img_recon_V)
            )
            Gradient_loss_I = L1Loss(
                kornia.filters.SpatialGradient()(data_IR),
                kornia.filters.SpatialGradient()(img_recon_I)
            )
            # Total loss
            #loss_I = mse_loss_I+ssim(data_IR,img_recon_I)+cc(img_recon_I,data_IR)+0.5*ms_ssim(img_recon_I,data_IR)
            #loss_V=mse_loss_V+ssim(data_IR,img_recon_V)+Gradient_loss_V+cc(img_recon_V,data_VIS)+2*ms_ssim(img_recon_V,data_VIS)

            print("mse_loss", type(huber_loss))
            #ss the weight parameters have
           # loss=weight[0]*huber_loss+weight[1]*(Gradient_loss_V+Gradient_loss_I)+weight[2]*(ssim(data_VIS,img_recon_V)+ssim(data_IR,img_recon_I))+weight[3]*(MSE_LOSS(img_recon_V.cpu(),data_VIS.cpu())+MSE_LOSS(img_recon_I.cpu(),data_IR.cpu()))+weight[4]*(cc(img_recon_V,data_VIS)+cc(img_recon_I,data_IR))
            #print('mse_v',huber(data_VIS,img_recon_V))
            loss=huber_loss+10*(Gradient_loss_V+Gradient_loss_I)+(ssim(data_VIS,img_recon_V)+ssim(data_IR,img_recon_I))+0.005*(MSE_LOSS(img_recon_V.cpu(),data_VIS.cpu())+MSE_LOSS(img_recon_I.cpu(),data_IR.cpu()))

            print('loss',loss)
            #print('mse_I', huber(data_IR, img_recon_I))
            print("mse_loss",huber_loss)
            #print('gradient_loss_V',Gradient_loss_V)
            print("gradient_loss",Gradient_loss_I+Gradient_loss_V)
            print('ssim_loss',(ssim(data_VIS,img_recon_V)+ssim(data_IR,img_recon_I)))
            #print('ssim_i', (ssim(data_IR, img_recon_I)))
           # print('MSE_L',MSE_LOSS(img_recon_V.cpu(),data_VIS.cpu()))
            print('MSE_LOSS', MSE_LOSS(img_recon_I.cpu(),data_IR.cpu())+MSE_LOSS(img_recon_V.cpu(),data_VIS.cpu()))
            print('cc',cc(img_recon_V,data_VIS)+cc(img_recon_I, data_IR))
            #print('cc_I', cc(img_recon_I, data_IR))
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            los = loss.item()
            Gradient_loss=Gradient_loss_V+Gradient_loss_I
            cc_1=cc(img_recon_V,data_VIS)+cc(img_recon_I,data_IR)
            ssim_1=ssim(data_VIS,img_recon_V)+ssim(data_IR,img_recon_I)
            MSE=MSE_LOSS(img_recon_V.cpu(),data_VIS.cpu())+MSE_LOSS(img_recon_I.cpu(),data_IR.cpu())
            print('Epoch/step: %d/%d, loss: %.7f, lr: %f' % (
            iteration + 1, step + 1, los, optimizer1.state_dict()['param_groups'][0]['lr']))
            # Save Loss
            loss_train.append(loss.item())
            huber_loss_train.append(huber_loss.item())
            Gradient_loss_train.append(Gradient_loss.item())
            cc_loss_train.append(cc_1.item())
            ssim_loss_train.append(ssim_1.item())
            ms_ssim_loss_train.append(MSE.item())

        scheduler1.step()
        scheduler2.step()

        lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])
        lr_list2.append(optimizer2.state_dict()['param_groups'][0]['lr'])
       # lr_list3.append(optimizer3.state_dict()['param_groups'][0]['lr'])


# Save Weights and result
torch.save({'weight': decoder.state_dict(), 'epoch': epochs},
           os.path.join(train_path, 'decoder44.pkl'))
torch.save({'weight': encoder.state_dict(), 'epoch': epochs},
           os.path.join(train_path, 'encoder44.pkl'))


scio.savemat(os.path.join(train_path, 'TrainData44.mat'),
                         {'Loss': np.array(loss_train),
                          'ssim_loss'  : np.array(ssim_loss_train),
                          'huber_loss': np.array(huber_loss_train),
                          'cc_loss': np.array(cc_loss_train),
                          'ms_ssim_loss':np.array(ms_ssim_loss_train),
                          'Gradient_loss': np.array(Gradient_loss_train),
                          })
scio.savemat(os.path.join(train_path, 'TrainData_plot_loss44.mat'),
                         {'loss_train': np.array(loss_train),
                          'ssim_loss_train': np.array(ssim_loss_train),
                          'huber_loss_train': np.array(huber_loss_train),
                          'cc_loss_train': np.array(cc_loss_train),
                          'ms_ssim_loss':np.array(ms_ssim_loss_train),
                          'Gradient_loss_train': np.array(Gradient_loss_train),
                          })

# plot
def Average_loss(loss):
    return [sum(loss[i * Iter_per_epoch:(i + 1) * Iter_per_epoch]) / Iter_per_epoch for i in range(int(len(loss) / Iter_per_epoch))]


plt.figure(figsize=[12,8])
plt.subplot(2,3,1), plt.plot(Average_loss(loss_train)), plt.title('Loss22')
plt.subplot(2,3,2), plt.plot(Average_loss(huber_loss_train)), plt.title('huber_loss22')
plt.subplot(2,3,3), plt.plot(Average_loss(ssim_loss_train)), plt.title('ssim_loss22')
plt.subplot(2,3,4), plt.plot(Average_loss(cc_loss_train)), plt.title('cc_loss22')
plt.subplot(2,3,5), plt.plot(Average_loss(ms_ssim_loss_train)), plt.title('ms_ssim_loss22')
plt.subplot(2,3,6), plt.plot(Average_loss(Gradient_loss_train)), plt.title('Gradient_loss22')
plt.tight_layout()

plt.savefig(os.path.join(train_path, 'curve_per_epoch44.png'))