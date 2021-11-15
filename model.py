import torch.nn as nn
import torch.nn.functional as F

class DepthWiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthWiseSepConv, self).__init__()
        
        #DIVIDE IN GROUPS AND CONV
        self.depthw_conv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,
                                 stride=stride,padding=padding,groups=in_channels,bias=False)
        
        #POINT WISE CONV AND CONCATENATE
        self.pointw_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,
                                 bias=False)        
    def forward(self, x):
        x = self.depthw_conv(x)
        x = self.pointw_conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=36, kernel_size=(3, 3), padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(36),
        ) # output_size = 32x32        
        
        # POOLING LAYER WITH STRIDE 2
        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride =(2,2),padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(36),
        ) # output_size = 16x16

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=70, kernel_size=(3, 3),padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(70),
        ) # output_size = 14x14        
        
       #POOLING LAYER WITH STRIDE 2
        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=70, out_channels=70, kernel_size=(3, 3), stride=(2,2),padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(70),
        ) # output_size = 7x7

        # CONVOLUTION BLOCK 3 (WITH DILATED CONVOLUTION)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=70, out_channels=128, kernel_size=(3, 3), padding=1,dilation=(2, 2),bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
        ) # output_size = 5x5  
        
        # CONVOLUTION BLOCK 4 (DEPTHWISE SEPERABLE CONVOLUTION)
        self.convblock4 = nn.Sequential(
            DepthWiseSepConv(in_channels=128,out_channels=256,kernel_size=(3,3) ),
            nn.ReLU(),            
            nn.BatchNorm2d(256),
        ) # output_size = 3x3
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) # output_size = 1

        self.fc1 = nn.Sequential(
            nn.Linear(256, 10,bias=False),
        )  


    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
                      
        x = self.convblock2(x)
        x = self.pool2(x)
        
        x = self.convblock3(x)

        x = self.convblock4(x)
        x = self.gap(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)
