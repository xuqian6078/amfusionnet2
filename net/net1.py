import torch
import torch.nn as nn
import torch.nn.functional as F
class conv3(nn.Module):
    def __init__(self,inplanes,outplanes,kernel_size=3,stride=1,padding=1):
        super(conv3,self).__init__()
        self.conv3=nn.Sequential(nn.Conv2d(inplanes,outplanes,kernel_size,stride,padding),
                         nn.ReLU())
    def forward(self,x):
        return self.conv3(x)
def conv5(inplanes,outplanes,kernel_size=5,stride=1,padding=0):
    return nn.Sequential(nn.Conv2d(inplanes,outplanes,kernel_size,stride,padding),
                         nn.ReLU())
def conv7(inplanes,outplanes,kernel_size=7,stride=1,padding=0):
    return nn.Sequential(nn.Conv2d(inplanes,outplanes,kernel_size,stride,padding),
                         nn.ReLU())
class Encoder(nn.Module):
    def __init__(self,inplanes,kernel_size=None,stride=1,padding=None):
        super(Encoder,self).__init__()
        out_channel=[32,64,128]
        self.conv3_1=nn.Sequential(
            nn.Conv2d(inplanes,out_channel[0],kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.conv3_2=nn.Sequential(
            nn.Conv2d(out_channel[0]*3,out_channel[1],kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.conv3_3=nn.Sequential(
            nn.Conv2d(out_channel[1]*3,out_channel[2],kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.conv5_1=nn.Sequential(
            nn.Conv2d(inplanes,out_channel[0],kernel_size=5,stride=1,padding=1),
            nn.ReLU()
        )
        self.conv5_2=nn.Sequential(
            nn.Conv2d(out_channel[0]*3,out_channel[1],kernel_size=5,padding=1),
            nn.ReLU()
        )
        self.conv5_3=nn.Sequential(
            nn.Conv2d(out_channel[1]*3,out_channel[2],kernel_size=5,padding=1),
            nn.ReLU()
        )
        self.conv7_1=nn.Sequential(
            nn.Conv2d(inplanes,out_channel[0],kernel_size=7,padding=1),
            nn.ReLU()
        )
        self.conv7_2=nn.Sequential(nn.Conv2d(out_channel[0]*3,out_channel[1],kernel_size=7,padding=1),nn.ReLU())
        self.conv7_3=nn.Sequential(
            nn.Conv2d(out_channel[1]*3,out_channel[2],kernel_size=7,padding=1),
            nn.ReLU()
        )
        #self.conv1=nn.Sequential(nn.Conv2d(out_channel[2]*3,out_channel[2],kernel_size=1,padding=0),nn.Relu())
        self.conv1=nn.Sequential(nn.Conv2d(inplanes,out_channel[2]*3,kernel_size=1,padding=0),nn.ReLU())
        self.conv1_=nn.Sequential(nn.Conv2d(out_channel[2]*3,out_channel[2],kernel_size=1,padding=0),nn.ReLU())
       #self.upsample=nn.UpsamplingBilinear2d(size=[224,224])
        self.conv3=conv3(1,16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shard_mlp=nn.Sequential(nn.Linear(384,384//16),
                                      nn.ReLU(),
                                      nn.Linear(384//16,384),
                                     nn.Sigmoid())
        self.sigmoid=nn.Sigmoid()
        self.spatial=nn.Sequential(
            nn.Conv2d(2,1,7,1,3),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
        self.ire=Iresidual()
        self.conv8=nn.Sequential(
            nn.Conv2d(384,768,1,1,padding=0),
            nn.PReLU())
    def forward(self,x):
        y=self.conv3(x)
        y3_1=self.conv3_1(y)
        y5_1=self.conv5_1(y)
        #y5_1=F.upsample(y5_1,size=(x.shape[2],x.shape[3]),mode='bilinear')
        y5_1 =nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(y5_1)
        y7_1=self.conv7_1(y)
        #y7_1=F.upsample(y7_1,size=(x.shape[2],x.shape[3]),mode='bilinear')
        y7_1=nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(y7_1)


        y3_2=self.conv3_2(torch.cat((y3_1,y5_1,y7_1),1))

        y5_2=self.conv5_2(torch.cat((y3_1,y5_1,y7_1),1))
        #y5_2=F.upsample(y5_2,size=(x.shape[2],x.shape[3]),mode='bilinear')
        y5_2=nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(y5_2)
        y7_2=self.conv7_2(torch.cat((y3_1,y5_1,y7_1),1))
        #y7_2=F.upsample(y7_2,size=(x.shape[2],x.shape[3]),mode='bilinear')
        y7_2=nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(y7_2)
        y3_3=self.conv3_3(torch.cat((y3_2,y5_2,y7_2),1))

        y5_3=self.conv5_3(torch.cat((y3_2,y5_2,y7_2),1))
        #y5_3=F.upsample(y5_3,size=(x.shape[2],x.shape[3]),mode='bilinear')
        y5_3=nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(y5_3)
        y7_3=self.conv7_3(torch.cat((y3_2,y5_2,y7_2),1))
        #y7_3=F.upsample(y7_3,size=(x.shape[2],x.shape[3]),mode='bilinear')
        y7_3=nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(y7_3)
        y1=torch.cat((y7_3,y5_3,y3_3),1)
        y2=self.avg_pool(y1)
        y2_b,y2_c,_,_=y2.size()
        y2=y2.view(y2_b,y2_c)
        y3=self.max_pool(y1)
        y3_b,y3_c,_,_=y3.size()
        y3=y3.view(y3_b,y3_c)
        y2=self.shard_mlp(y2).view(y2_b,y2_c,1,1)
        y3=self.shard_mlp(y3).view(y3_b,y3_c,1,1)
        y_channel=self.sigmoid(y2+y3)
        y_spatial=torch.cat((torch.max(y1,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)
        y_spatial=self.spatial(y_spatial)
        y_spatial=self.sigmoid(y_spatial)
        y_spatial_1=y_spatial*y1
        y_channel_1=y_channel*y1
        y_sp_ch=torch.cat((y_spatial_1,y_channel_1),1)
        y8=self.conv8(y1)
        y9=y_sp_ch+y8
        #y_zong=y_spatial_1+y_channel_1
        #y_zong=self.sigmoid(y_zong)

        #y_ire=self.ire(y_sp_ch)
        #y=y1+y_ire
        #spatial attention
        #y2=self.conv1(y)
        #y=self.conv1_(y1+y2)
        return y1,y_channel,y_spatial,y9
class Iresidual(nn.Module):
    def __init__(self):
        super(Iresidual,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(768,768,3,1,1,bias=False),
            nn.GELU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(768,384,1,1,bias=False),
            nn.GELU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(384,384,1,1,bias=False),
            nn.BatchNorm2d(384),
        )
    def forward(self,x):
        residual=x
        y=self.conv1(x)
        y=residual+y
        y=self.conv2(y)
        y=self.conv3(y)
        return y

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(768,384,3,1,1),
                                 nn.BatchNorm2d(384),
                                 nn.LeakyReLU())
        self.conv2=nn.Sequential(nn.Conv2d(384,192,3,1,1),
                                 nn.BatchNorm2d(192),
                                 nn.LeakyReLU())
        self.conv3=nn.Sequential(nn.Conv2d(192,1,3,1,1),
                                 nn.LeakyReLU())
        '''
        self.conv4= nn.Sequential(nn.Conv2d(96,48, 3, 1, 1),
                                  nn.BatchNorm2d(48),
                                   nn.LeakyReLU())
        self.conv5= nn.Sequential(nn.Conv2d(48, 1, 3, 1, 1),
                                   nn.LeakyReLU())
        '''
    def forward(self,x):
        y=self.conv1(x)
        y=self.conv2(y)
        y=self.conv3(y)
        return y






