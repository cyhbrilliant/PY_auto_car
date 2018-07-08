#--------------------------------------------------------------------------------
#   Based on Efficient ConvNet for Real-time Semantic Segmentation
#   deconv->nnupsample+conv
#   input 2:1 BGR
#   output 2:1 pixel-softmax
#   2017.12.31   18.45
#   Memory Using:   5905MB
#   Speed:  1 batch of 12 per second
#   Value:  useful
#   Torch version
#--------------------------------------------------------------------------------
from AutoCar.Model_Segment import labels2 as lb
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import h5py

class Downsample_block(torch.nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.MaxPool=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels-in_channels,
                                  kernel_size=3,stride=2,padding=1)
    def forward(self, x):
        return F.relu(torch.cat((self.Conv(x),self.MaxPool(x)),1))
#-----Test Downsample_block
# D=Downsample_block(1,2)
# x=D(torch.autograd.Variable(torch.rand(1,1,10,10)))

class Res_block(torch.nn.Module):
    def __init__(self,channel,dilatedRate,dropoutKeepProb):
        super(Res_block, self).__init__()
        self.Bn1=torch.nn.BatchNorm2d(num_features=channel)
        self.Bn2=torch.nn.BatchNorm2d(num_features=channel)
        self.Conv3_1=torch.nn.Conv2d(in_channels=channel,out_channels=channel,
                                     kernel_size=(3,1),stride=1,padding=(1,0))
        self.Conv1_3=torch.nn.Conv2d(in_channels=channel,out_channels=channel,
                                     kernel_size=(1,3),stride=1,padding=(0,1))
        self.ConvDilate3_1=torch.nn.Conv2d(in_channels=channel,out_channels=channel,
                                     kernel_size=(3,1),stride=1,padding=(dilatedRate,0),dilation=dilatedRate)
        self.ConvDilate1_3=torch.nn.Conv2d(in_channels=channel,out_channels=channel,
                                     kernel_size=(1,3),stride=1,padding=(0,dilatedRate),dilation=dilatedRate)
        self.Dropout=torch.nn.Dropout2d(p=dropoutKeepProb)
    def forward(self, x):
        x1=F.relu(self.Conv3_1(x))
        x1=F.relu(self.Bn1(self.Conv1_3(x1)))
        x1=F.relu(self.ConvDilate3_1(x1))
        x1=self.Dropout(F.relu(self.Bn2(self.ConvDilate1_3(x1))))
        x=F.relu(x+x1)
        return x
# #-----Test Res_block
# D=Res_block(5,8,5,True,0.3)
# x=D(torch.autograd.Variable(torch.rand(10,5,20,20)))

class Upsample_block(torch.nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.Upsample=torch.nn.Upsample(scale_factor=2,mode='nearest')
        self.Conv=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        return F.relu(self.Conv(self.Upsample(x)))
#-----Test Upsample_block
# D=Upsample_block(1,2)
# x=D(torch.autograd.Variable(torch.rand(1,1,10,10)))
# print(x.size())

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Downsample_block_1=Downsample_block(3,16)
        self.Downsample_block_2=Downsample_block(16,64)
        self.Res_block_1_1=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Res_block_1_2=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Res_block_1_3=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Res_block_1_4=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Res_block_1_5=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Downsample_block_3=Downsample_block(64,128)
        self.Res_block_2_1=Res_block(channel=128,dilatedRate=2,dropoutKeepProb=0.3)
        self.Res_block_2_2=Res_block(channel=128,dilatedRate=4,dropoutKeepProb=0.3)
        self.Res_block_2_3=Res_block(channel=128,dilatedRate=6,dropoutKeepProb=0.3)
        self.Res_block_2_4=Res_block(channel=128,dilatedRate=8,dropoutKeepProb=0.3)
        self.Res_block_2_5=Res_block(channel=128,dilatedRate=2,dropoutKeepProb=0.3)
        self.Res_block_2_6=Res_block(channel=128,dilatedRate=4,dropoutKeepProb=0.3)
        self.Res_block_2_7=Res_block(channel=128,dilatedRate=6,dropoutKeepProb=0.3)
        self.Res_block_2_8=Res_block(channel=128,dilatedRate=8,dropoutKeepProb=0.3)
        self.Upsample_block_1=Upsample_block(128,64)
        self.Res_block_3_1=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Res_block_3_2=Res_block(channel=64,dilatedRate=1,dropoutKeepProb=0.3)
        self.Upsample_block_2=Upsample_block(64,16)
        self.Res_block_4_1=Res_block(channel=16,dilatedRate=1,dropoutKeepProb=0.3)
        self.Res_block_4_2=Res_block(channel=16,dilatedRate=1,dropoutKeepProb=0.3)
        self.Upsample_block_3=Upsample_block(16,29)
        # self.Softmax=torch.nn.Softmax2d()
    def forward(self,x):
        x=self.Downsample_block_1(x)
        x=self.Downsample_block_2(x)
        x=self.Res_block_1_1(x)
        x=self.Res_block_1_2(x)
        x=self.Res_block_1_3(x)
        x=self.Res_block_1_4(x)
        x=self.Res_block_1_5(x)
        x=self.Downsample_block_3(x)
        x=self.Res_block_2_1(x)
        x=self.Res_block_2_2(x)
        x=self.Res_block_2_3(x)
        x=self.Res_block_2_4(x)
        x=self.Res_block_2_5(x)
        x=self.Res_block_2_6(x)
        x=self.Res_block_2_7(x)
        x=self.Res_block_2_8(x)
        x=self.Upsample_block_1(x)
        x=self.Res_block_3_1(x)
        x=self.Res_block_3_2(x)
        x=self.Upsample_block_2(x)
        x=self.Res_block_4_1(x)
        x=self.Res_block_4_2(x)
        x=self.Upsample_block_3(x)
        # x=self.Softmax(x)
        return x

def Predict(Inputimg, IsGpuEval):
    net = Net()
    if IsGpuEval:
        net.load_state_dict(torch.load('Model_Segment/normal_1_Hist_320_3.pkl'))  # 载入参数
        net.eval()
        net.cuda()
    else:
        net.load_state_dict(torch.load('Model_Segment/normal_1_Hist_320_3.pkl', map_location=lambda storage, loc: storage))  # 载入参数
        net.eval()

    # Inputimg = cv2.resize(Inputimg, dsize=(320, 160))
    Inputimg = np.concatenate((cv2.equalizeHist(Inputimg[:, :, 0])[:, :, np.newaxis],
                               cv2.equalizeHist(Inputimg[:, :, 1])[:, :, np.newaxis],
                               cv2.equalizeHist(Inputimg[:, :, 2])[:, :, np.newaxis]), axis=2)
    Data=np.transpose(Inputimg,axes=(2,0,1))[np.newaxis,:]
    Input=torch.autograd.Variable(torch.from_numpy(Data.astype(np.float32)))
    if IsGpuEval:
        Input=Input.cuda()
    Predict=net(Input)
    if IsGpuEval:
        Predict=Predict.cpu().data.numpy()
    else:
        Predict=Predict.data.numpy()
    PredictArgmax=np.argmax(Predict,axis=1)[0,:]
    # print(Predict)
    # print('ok')
    PredictPic = np.zeros([PredictArgmax.shape[0], PredictArgmax.shape[1], 3], dtype=np.uint8)
    for i in range(PredictArgmax.shape[0]):
        for j in range(PredictArgmax.shape[1]):
            PredictPic[i,j,0]=lb.labels[PredictArgmax[i,j]].color[2]
            PredictPic[i,j,1]=lb.labels[PredictArgmax[i,j]].color[1]
            PredictPic[i,j,2]=lb.labels[PredictArgmax[i,j]].color[0]
    return PredictPic, PredictArgmax
    # cv2.imshow('Predict',PredictPic)
    # cv2.waitKey(0)
