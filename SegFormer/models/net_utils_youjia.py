import torch.nn as nn
import torch
from torch.nn import functional as F
import math


class Fusionmodel(nn.Module):
    def __init__(self, in_channels, patch_h, patch_w):
        super(Fusionmodel, self).__init__()
        self.n_h, self.n_w = patch_h,patch_w
        self.seen = 0
        self.channels = in_channels
        self.out_channels = in_channels //2

        self.rgb_msc = MSC(in_channels)
        self.t_msc = MSC(in_channels)

        self.RGB_key = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.RGB_query = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.RGB_value = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.RGB_W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.channels,
                           kernel_size=1, stride=1, padding=0)

        self.T_key = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.T_query = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.T_value = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.T_W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.channels,
                           kernel_size=1, stride=1, padding=0)


        self.gate_RGB = nn.Conv2d(self.channels * 2, 1, kernel_size=1, bias=True)
        self.gate_T = nn.Conv2d(self.channels * 2, 1, kernel_size=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.gate_fusion = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=True)

    def forward(self, RGB, T):

        # pyramid pooling
        RGB_m = self.rgb_msc(RGB)# RGB pyramid 上下文特征，(1,out_channels,128,128)
        T_m = self.t_msc(T)# T上下文特征，It, Contextual Information Extraction:对图像特征进行多尺度max_pooling拼接得到上下文信息

        # # Only pyramid
        # new_shared = (RGB_m + T_m)/2
        # return RGB_m, T_m, new_shared

        # # no pyramid pooling
        # RGB_m = RGB
        # T_m = T


        # # original across non-local attention
        # batch_size, channels = RGB_m.size(0), RGB_m.size(1) // 2
        #
        # rgb_query = self.RGB_query(RGB_m).view(batch_size, channels, -1)# (batch_size, 19, -1)
        # rgb_query = rgb_query.permute(0, 2, 1)#(1,16384,19)
        # rgb_key = self.RGB_key(RGB_m).view(batch_size, channels, -1)#(1,19,-1)
        # rgb_value = self.RGB_value(RGB_m).view(batch_size, channels, -1).permute(0, 2, 1)#(1,-1,19)
        # T_query = self.T_query(T_m).view(batch_size, channels, -1)
        # T_query = T_query.permute(0, 2, 1)
        # T_key = self.T_key(T_m).view(batch_size, channels, -1)
        # T_value = self.T_value(T_m).view(batch_size, channels, -1).permute(0, 2, 1)
        #
        # RGB_sim_map = torch.matmul(T_query, rgb_key)  # (1,-1,-1)
        # RGB_sim_map = (channels ** -.5) * RGB_sim_map
        # RGB_sim_map = F.softmax(RGB_sim_map, dim=-1)
        # RGB_context = torch.matmul(RGB_sim_map, rgb_value)  # (1,-1,38)
        # RGB_context = RGB_context.permute(0, 2, 1).contiguous()
        # RGB_context = RGB_context.view(batch_size, channels, *RGB_m.size()[2:])  # (1,19,128,128)
        # RGB_context = self.RGB_W(RGB_context)
        #
        # T_sim_map = torch.matmul(rgb_query, T_key)
        # T_sim_map = (channels ** -.5) * T_sim_map
        # T_sim_map = F.softmax(T_sim_map, dim=-1)
        # T_context = torch.matmul(T_sim_map, T_value)
        # T_context = T_context.permute(0, 2, 1).contiguous()
        # T_context = T_context.view(batch_size, channels, *T_m.size()[2:])
        # T_context = self.T_W(T_context)


        # # non-local with spacial partition in the same modality
        # feature_size, channels = RGB_m.size()[2:], RGB_m.size(1) // 2
        # rgb_query, _ = self.spacial_split(self.RGB_query(RGB_m))
        # rgb_query = rgb_query.permute(0, 2, 1)
        # rgb_key, patch_param = self.spacial_split(self.RGB_key(RGB_m))
        # rgb_value, _ = self.spacial_split(self.RGB_value(RGB_m))
        # rgb_value = rgb_value.permute(0, 2, 1)
        # T_query, _ = self.spacial_split(self.T_query(T_m))
        # T_query = T_query.permute(0, 2, 1)  # [BP, N, C]
        # T_key, _ = self.spacial_split(self.T_key(T_m))  # [BP, C, N]
        # T_value, _ = self.spacial_split(self.T_value(T_m))
        # T_value = T_value.permute(0, 2, 1)  # [BP, N, C]
        #
        # RGB_sim_map = torch.matmul(rgb_query, rgb_key)  # [BP, N, N]
        # RGB_sim_map = (channels ** -.5) * RGB_sim_map
        # RGB_sim_map = F.softmax(RGB_sim_map, dim=-1)
        # RGB_context = torch.matmul(RGB_sim_map, rgb_value)  # [BP, N, C]
        # RGB_context = self.spacial_splice(RGB_context, patch_param)
        # RGB_context = self.RGB_W(RGB_context)
        #
        # T_sim_map = torch.matmul(T_query, T_key)
        # T_sim_map = (channels ** -.5) * T_sim_map
        # T_sim_map = F.softmax(T_sim_map, dim=-1)
        # T_context = torch.matmul(T_sim_map, T_value)
        # T_context = self.spacial_splice(T_context, patch_param)
        # T_context = self.T_W(T_context)

        # across non-local with spacial partition
        feature_size, channels = RGB_m.size()[2:], RGB_m.size(1) // 2
        rgb_query, _ = self.spacial_split(self.RGB_query(RGB_m))
        rgb_query = rgb_query.permute(0, 2, 1)
        rgb_key, patch_param = self.spacial_split(self.RGB_key(RGB_m))
        rgb_value, _ = self.spacial_split(self.RGB_value(RGB_m))
        rgb_value = rgb_value.permute(0, 2, 1)
        T_query, _ = self.spacial_split(self.T_query(T_m))
        T_query = T_query.permute(0, 2, 1)#[BP, N, C]
        T_key, _ = self.spacial_split(self.T_key(T_m))#[BP, C, N]
        T_value, _ = self.spacial_split(self.T_value(T_m))
        T_value = T_value.permute(0, 2, 1)#[BP, N, C]


        RGB_sim_map = torch.matmul(T_query, rgb_key)#[BP, N, N]
        RGB_sim_map = (channels ** -.5) * RGB_sim_map
        RGB_sim_map = F.softmax(RGB_sim_map, dim=-1)
        RGB_context = torch.matmul(RGB_sim_map, rgb_value)#[BP, N, C]
        RGB_context = self.spacial_splice(RGB_context, patch_param)
        RGB_context = self.RGB_W(RGB_context)

        T_sim_map = torch.matmul(rgb_query, T_key)
        T_sim_map = (channels ** -.5) * T_sim_map
        T_sim_map = F.softmax(T_sim_map, dim=-1)
        T_context = torch.matmul(T_sim_map, T_value)
        T_context = self.spacial_splice(T_context, patch_param)
        T_context = self.T_W(T_context)

        # # only NL
        # new_RGB = (RGB + RGB_context) / 2  # RGB_i_out
        # new_T = (T + T_context) / 2  # T_i_out
        #
        # new_shared = (new_RGB + new_T)/2
        #
        # return new_RGB, new_T, new_shared

        # # Pyramid + NL
        # new_RGB = (RGB + RGB_m + RGB_context) / 3  # RGB_i_out
        # new_T = (T + T_m + T_context) / 3  # T_i_out
        #
        # new_shared = (new_RGB + new_T)/2
        #
        # return new_RGB, new_T, new_shared

        #
        # # no non-local attention
        # cat_fea = torch.cat([RGB_m, T_m], dim=1)

        # # no non-local, no pyramid pooling
        # cat_fea = torch.cat([RGB, T], dim=1)

        ''' Information Aggregation '''
        # cat_fea = torch.cat([T_context, RGB_context], dim=1)

        # attention_vector_RGB = self.gate_RGB(cat_fea)
        # attention_vector_T = self.gate_T(cat_fea)

        # attention_vector = torch.cat([attention_vector_RGB, attention_vector_T], dim=1)
        # attention_vector = self.softmax(attention_vector)  # attention vector in paper
        # attention_vector_RGB, attention_vector_T = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        # new_shared = RGB * attention_vector_RGB + T * attention_vector_T  # feature aggretation

        # new_RGB = (RGB + new_shared) / 2  # RGB_i_out
        # new_T = (T + new_shared) / 2  # T_i_out

        # new_RGB = self.relu1(new_RGB)
        # new_T = self.relu2(new_T)

        # merge = torch.cat((new_RGB, new_T), dim=1)
        # fusion = self.gate_fusion(merge)
        # # return new_RGB, new_T, new_shared

        ''' 단순 concat '''
        new_RGB = (RGB + RGB_context) / 2
        new_T = (T + T_context) / 2
        fusion = (new_RGB + new_T) / 2

        return new_RGB, new_T, fusion

    def spacial_split(self, Fea):
        batch, channels, H, W = Fea.shape
        num_patches = self.n_h*self.n_w
        new_H = int(math.ceil(H / self.n_h) * self.n_h)
        new_W = int(math.ceil(W / self.n_w) * self.n_w)

        interpolate = False
        if new_H != H or new_W != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            Fea = F.interpolate(Fea, size=(new_H, new_W), mode="bilinear", align_corners=False)
            interpolate = True
        patch_h = new_H //self.n_h
        patch_w = new_W //self.n_w
        patch_unit = patch_h * patch_w


        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_Fea = Fea.reshape(batch * channels * self.n_h, patch_h, self.n_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_Fea = reshaped_Fea.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_Fea = transposed_Fea.reshape(batch, channels, num_patches, patch_unit)
        # [B, C, N, P] --> [B, P, C, N]
        transposed_Fea = reshaped_Fea.permute(0, 3, 1, 2)
        # [B, P, C, N] --> [BP, C, N]
        patches = transposed_Fea.reshape(batch * patch_unit, channels, -1)

        # # [B, P, C, N] --> [B, PC, N]
        # patches = transposed_Fea.reshape(batch, patch_unit * channels, -1)

        return patches, [batch, channels, H, W, patch_h, patch_w]

    def spacial_splice(self, patches, patch_param):
        [batch_size, channels, H, W, patch_h, patch_w] = patch_param
        patch_unit = patch_h * patch_w
        num_patches = self.n_h*self.n_w

        # # [B, N, PC] --> [B, N, P, C]
        # patches = patches.reshape(batch_size, num_patches, patch_unit, -1)
        # # [B, P, N, C] --> [B, C, N, P]
        # patches = patches.permute(0, 3, 1, 2).contiguous()

        # [BP, N, C] --> [B, P, N, C]
        patches = patches.reshape(batch_size, patch_unit, num_patches, -1)
        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.permute(0, 3, 2, 1).contiguous()

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        Fea = patches.reshape(batch_size * channels * self.n_h, self.n_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        Fea = Fea.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, new_H, new_W]
        Fea = Fea.reshape(batch_size, channels, self.n_h * patch_h, self.n_w * patch_w)
        if self.n_h * patch_h != H or self.n_w * patch_w != W:
            Fea = F.interpolate(Fea, size=(H, W), mode="bilinear", align_corners=False)
        return Fea
# # Contextual Information Extraction:conv 1*1(cat(不同等级的2^l的max-pooling)
class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion

def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
