import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
import math


class Fusionmodel(nn.Module):
    def __init__(self, in_channels):
        super(Fusionmodel, self).__init__()
        self.seen = 0

        self.channels = in_channels
        self.out_channels = in_channels // 2

        self.RGB = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                kernel_size=1, stride=1, padding=0)
        self.D = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                kernel_size=1, stride=1, padding=0)

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.value = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.channels,
                           kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channels*2, out_channels=self.channels,
                                kernel_size=1, stride=1, padding=0)

        self.gate_RGB = nn.Conv2d(self.channels * 2, 1, kernel_size=1, bias=True)
        self.gate_D = nn.Conv2d(self.channels * 2, 1, kernel_size=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.gate_fusion = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=True)

    def forward(self, RGB, D):

        # ''' only IIA '''
        # ''' naive ours CWF change (channel-wise feature aggregation design change) '''
        
        # # RGB = (8, 64, 120, 160)
        # # D = (8, 64, 120, 160)
        # batch_size = RGB.size(0)
        # channel = RGB.size(1)
        # H = RGB.size(2)
        # W = RGB.size(3)

        # '''RGB, Depth 각각 (NxC) 로 만들어주기'''
        # RGB_m = RGB.view(batch_size, channel, -1, 1)    # (B, C, HW, 1)
        # D_m = D.view(batch_size, channel, -1, 1)

        # concat = torch.cat([RGB_m, D_m], dim=2) # (B, C, 2(HW), 1)

        # '''intra & inter non-local attention'''
        # channels = concat.size(1)

        # query  = self.query(concat).view(batch_size, channels, -1)
        # query = query.permute(0, 2, 1)             # (B, HW, C)
        # # print("query: ", query.size())             
        # key = self.key(concat).view(batch_size, channels, -1) # (B, C, HW)
        # # print("key: ", key.size())                 
        # value  = self.value(concat).view(batch_size, channels, -1).permute(0, 2, 1) # (B, HW, C)
        # # print("value: ", value.size())             

        # sim_map = torch.matmul(query, key)   # (B, HW, HW)
        # # print("sim_map: ", sim_map.size())
        # sim_map = (channels ** -.5) * sim_map
        # sim_map = F.softmax(sim_map, dim=-1)
        # context = torch.matmul(sim_map, value)
        # context = context.permute(0, 2, 1).contiguous() # (B, C, HW)
        # # print("context: ", context.size())
        # context = context.view(batch_size, channels//2, *concat.size()[2:]) # (B, C, HW)
        # context = self.W(context)                 # (B, C, 2(HW), 1)

        # '''RGB, D reshape'''
        # RGB_context = self.conv1(context)   # (B, C/2, HW, 1)
        # RGB_context = RGB_context.view(batch_size, channels, H, W)  # (B, C, H, W)
        # D_context = self.conv1(context)
        # D_context = D_context.view(batch_size, channels, H, W)

        # ''' 단순 concat '''
        # new_RGB = (RGB + RGB_context) / 2
        # new_D = (D + D_context) / 2
        # fusion = (new_RGB + new_D) / 2

        # return new_RGB, new_D, fusion


        ''' only CAF '''
        '''Information Aggregation'''
        cat_fea = torch.cat([RGB, D], dim=1)  # (B, 2C, H, W)
        cat_fea = self.conv2(cat_fea)   # (B, C, H, W)

        ''' GAP + adpative fusion '''
        attention_vector = self.GAP(cat_fea)                # (B, C, 1, 1)
        attention_vector = self.sigmoid(attention_vector)
        attention_vector_RGB = RGB * attention_vector      # (B, C, H, W)
        attention_vector_D = D * (1.0-attention_vector)

        ''' F_RGB + F'_RGB, D_RGB + D'_RGB '''
        new_RGB = RGB + attention_vector_RGB    # (B, C, H, W)
        new_D = D + attention_vector_D

        new_RGB = self.relu1(new_RGB)
        new_D = self.relu2(new_D)

        ''' element-wise multiplication '''
        fusion = torch.mul(new_RGB, new_D)

        return new_RGB, new_D, fusion


class Addmodel(nn.Module):
    def __init__(self, in_channels):
        super(Addmodel, self).__init__()
        self.seen = 0

        self.channels = in_channels
        self.out_channels = in_channels // 2

    def forward(self, RGB, D):
        fusion = (RGB + D) / 2

        return RGB, D, fusion
