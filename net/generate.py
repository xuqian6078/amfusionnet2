import torch
from net import Encoder,Decoder


import numpy as np
import torch
import os
from PIL import Image
from skimage.io import imsave
from util import Test_fusion


# =============================================================================
# Test Details
# =============================================================================
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
addition_mode='Sum'#'Sum'&'Average'&'l1_norm'
Test_data_choose='Test_data_TNO'#'Test_data_TNO'&'Test_data_NIR_Country'&'Test_data_FLIR'

if Test_data_choose=='Test_data_TNO':
    test_data_path = './Datasets/Test_data_TNO'
elif Test_data_choose=='Test_data_NIR_Country':
    test_data_path = '.\\Datasets\\Test_data_NIR_Country'
elif Test_data_choose=='Test_data_FLIR':
    test_data_path = '.\\Datasets\\Test_data_FLIR\\'

# Determine the number of files
Test_Image_Number=len(os.listdir(test_data_path))

# =============================================================================
# Test
# =============================================================================
for i in range(int(Test_Image_Number/2)):
    if Test_data_choose=='Test_data_TNO':
        Test_IR = Image.open(test_data_path+'/IR'+str(i+1)+'.bmp') # infrared image
        Test_Vis = Image.open(test_data_path+'/VIS'+str(i+1)+'.bmp') # visible image
    elif Test_data_choose=='Test_data_NIR_Country':
        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.png') # infrared image
        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.png') # visible image
    elif Test_data_choose=='Test_data_FLIR':
        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.jpg') # infrared image
        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.jpg') # visible image
    Fusion_image=Test_fusion(Test_IR,Test_Vis,addition_mode='l1_norm')
    imsave('./test_result/l1_norm23/F'+str(i+1)+'.png',Fusion_image)