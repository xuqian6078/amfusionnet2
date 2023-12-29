# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""
import numpy as np
import torch
from net import Encoder,Decoder
import torch.nn.functional as F
import cv2
import torch.nn as nn
from torch.autograd import Variable
import math
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPSILON=1e-10



def output_img(x):
    return x.cpu().detach().numpy()[0, 0, :, :]

def cc(imag1,imag2):
    #imag1.to("cuda")
    #imag2.to("cuda")
    I2=imag1.pow(2)
    J2=imag2.pow(2)
    IJ= imag1 * imag2
    I_ave=torch.mean(imag1)
    J_ave = torch.mean(imag2)
    I2_ave=torch.mean(I2)
    J2_ave=torch.mean(J2)
    IJ_ave=torch.mean(IJ)
    I_var=I2_ave-I_ave.pow(2)
    J_var=J2_ave-J_ave.pow(2)
     # imag1_std=torch.std(imag1)
     # imag2_std=torch.std(imag2)
    cross=IJ_ave-I_ave*J_ave
     #IJ_ave=torch.mean(IJ)
   #  s=imag1-I_ave
   #  g=imag2-J_ave
   #  s2=s**2
   #  g2=g**2
   #  IJ=imag1*imag2
   #  IJ_ave=torch.mean(IJ)
   #   cross=IJ_ave-I_ave*J_ave
   #   I_var=torch.var(imag1,unbiased=False)
   #   J_var=torch.var(imag2,unbiased=False)
    cc=cross/(torch.sqrt(I_var)*torch.sqrt(J_var)+1e-5)

    #cc=cross/(torch.sqrt(I_var)*torch.sqrt(J_var)+1e-5)

    # I_var=I2_ave-I_ave.pow(2)
    # J_var=J2_ave-J_ave.pow(2)
    #cc=torch.mean(s*g)/(torch.sqrt(torch.mean((s2))))*(torch.sqrt(torch.mean((g2))))
    return (1-cc)
    #sd2=imag2-torch.mean(imag2)
    #cc=torch.sum(sd1*sd2)/(torch.sqrt(torch.sum(sd1**2))*torch.sqrt(torch.sum(sd2**2)))
    #return (1-cc)

# def cc(A,F):
#
#     #A=A.detach().numpy()
#    # F=F.detach().numpy()
#     rAF=torch.sum((A-torch.mean(A))*(F-torch.mean(F)))/torch.sqrt(torch.sum((A-torch.mean(A))**2)*torch.sum((F-torch.mean(F))**2))
#     print(rAF)
#
#     #rBF=np.sum((B-np.mean(B))*(F-np.mean(F)))/np.sqrt(np.sum((B-np.mean(B))**2)*np.sum((F-np.mean(F))**2))
#     #CC=np.mean([rAF,rBF])
#     return (1-rAF)

def l1_addition(y1, y2, window_width=1):
    ActivityMap1 = y1.abs()
    ActivityMap2 = y2.abs()
    print('y1', y1.shape)
    # print('y2', y2.shape)
    kernel = torch.ones(2 * window_width + 1, 2 * window_width + 1) / (2 * window_width + 1) ** 2
    kernel = kernel.to(device).type(torch.float32)[None, None, :, :]
    kernel = kernel.expand(y1.shape[1], y1.shape[1], 2 * window_width + 1,
                           2 * window_width + 1)  # 生成y.shape[1]种，y.shape[1]通道的尺寸为3*3的滤波器
    # print('kernel',kernel)
    ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
    ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
    WeightMap1 = ActivityMap1 / (ActivityMap1 + ActivityMap2)
    WeightMap2 = ActivityMap2 / (ActivityMap1 + ActivityMap2)
    return WeightMap1 * y1 + WeightMap2 * y2


def Test_fusion(img_test1, img_test2,addition_mode=None):
    decoder = Decoder().to(device)
    encoder=Encoder(16).to(device)
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    encoder.load_state_dict(torch.load(
        "./Train_result/" + "31.pkl"
    )['weight'])
    decoder.load_state_dict(torch.load("./Train_result/" + "decoder31.pkl")['weight'])


    img_test1 = np.array(img_test1, dtype='float32') / 255  # 将其转换为一个矩阵
    #img_test1 = cv2.cvtColor(img_test1, cv2.COLOR_BGR2GRAY)
    print('IMAGE_TEST1', img_test1.shape)
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    print('img_test1shape',img_test1.shape)
    img_test2 = np.array(img_test2, dtype='float32') / 255  # 将其转换为一个矩阵
   # img_test2=cv2.cvtColor(img_test2,cv2.COLOR_BGR2GRAY)
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))
    print('img_test2shape', img_test2.shape)
    img_test1 = img_test1.to(device)
    img_test2 = img_test2.to(device)

    with torch.no_grad():
        F_I= encoder(img_test1)
        F_V= encoder(img_test2)
    if addition_mode=="softmax":
        f_i_min,_=torch.min(F_I,dim=1,keepdim=True)
        f_i_max,_=torch.max(F_I,dim=1,keepdim=True)

        f_v_min, _ = torch.min(F_V, dim=1, keepdim=True)
        f_v_max, _ = torch.max(F_V, dim=1, keepdim=True)

        f_i_map=(F_I-f_i_min)/(f_i_max-f_i_min)
        f_v_map=(F_V-f_v_min)/(f_v_max-f_v_min)

        w_i=torch.exp(f_i_map)/(torch.exp(f_i_map)+torch.exp(f_v_map)+EPSILON)
        w_v=torch.exp(f_v_map)/(torch.exp(f_i_map)+torch.exp(f_v_map)+EPSILON)

        f=w_i*F_I+w_v*F_V


    if addition_mode=="l1_norm":

        f=l1_addition(F_I,F_V)


    if addition_mode=="addition":
        f=(F_I+F_V)/2
    if addition_mode=="channel_sum":
        wi,_=torch.max(F_I,1)
        wv,_=torch.max(F_V,1)
        wi=wi/(wi+wv)
        wv=wv/(wi+wv)
        f=F_I*wi+F_V*wv

        #f=torch.cat((F_I,F_V),1)

    with torch.no_grad():
        Out = decoder(f)


    return output_img(Out)


def calcentropy(img):
    tmp = []
    e = 0

    # img.cpu().numpy()
    # img=cv2.imread(img)
    # img.Tensor.cpu()
    # img=img.cpu().detach.numpy()
    # a,b,c,d=img.shape
    # for i in range(a):
    entropy = []
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # print((img.shape))
    total_pixel = img.shape[0] * img.shape[1]
    for item in hist:
        probability1 = item / total_pixel
        if probability1 == 0:
            en = 0
        else:
            en = -1 * probability1 * (np.log(probability1) / np.log(2))
        entropy.append(en)
    res = np.sum(entropy)
    return res
