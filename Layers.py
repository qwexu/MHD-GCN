import torch
import torch.nn as nn
import scipy.io as io
import numpy as np
from RGNN import NewSGConv
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module): 
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class TemporalSpatialBlock(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(TemporalSpatialBlock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class MultiLevelSpectralBlock(nn.Module):
    def __init__(self, inc, params_path='./scaling_filter.mat'):
        super(MultiLevelSpectralBlock, self).__init__()
        self.filter_length = io.loadmat(params_path)['Lo_D'].shape[1]
        self.conv = nn.Conv2d(in_channels=inc,
                              out_channels=inc * 2,
                              kernel_size=(1, self.filter_length),
                              stride=(1, 2), padding=0,
                              groups=inc,
                              bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = io.loadmat(params_path)
                Lo_D, Hi_D = np.flip(f['Lo_D'], axis=1).astype('float32'), np.flip(f['Hi_D'], axis=1).astype('float32')
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis=0)).unsqueeze(1).unsqueeze(1).repeat(
                    inc, 1, 1, 1)
                m.weight.requires_grad = False

    def self_padding(self, x):
        return torch.cat((x[:, :, :, -(self.filter_length // 2 - 1):], x, x[:, :, :, 0:(self.filter_length // 2 - 1)]),
                         (self.filter_length // 2 - 1))

    def forward(self, x):
        out = self.conv(self.self_padding(x))
        return out[:, 0::2, :, :], out[:, 1::2, :, :]


class GlobalAdjLeaningLayer(nn.Module):
    def __init__(self, num_nodes):
        super(GlobalAdjLeaningLayer, self).__init__()
        self.num_nodes = num_nodes
        self.edge_weights = torch.rand([num_nodes, num_nodes])
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = self.edge_weights[self.xs, self.ys]  # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=True)

    def forward(self, mask):
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device="cuda")
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(
            edge_weight.diagonal())  # copy values from lower tri to upper tri
        edge_weight = edge_weight * mask
        edge_weight = edge_weight.reshape(-1)
        return edge_weight


class GcnBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super(GcnBlock, self).__init__()
        self.conv = NewSGConv(in_c, out_c, 5)
        self.bacthnorm = nn.BatchNorm1d(out_c, momentum=0.1, affine=True, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.out_c = out_c
        self.elu = nn.ELU(inplace=True)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.bacthnorm.reset_parameters()

    def forward(self, x, edge, weights):
        out = self.conv(x, edge, weights)
        out = out.transpose(1, 2)
        out = self.bacthnorm(out)
        out = out.transpose(1, 2)
        out = self.dropout(out)

        return out


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.flattetn = nn.Flatten(1, 2)
        self.lin1 = nn.Linear(64 * 2, 32)
        self.bn = nn.BatchNorm1d(32)
        self.lin2 = nn.Linear(32, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.flattetn(x)
        out = self.lin1(out)
        out = self.lin2(self.bn(out))
        out = self.softmax(out)
        return out


class MHDGcnNet(nn.Module):
    def __init__(self, si, dropout):
        super(MHDGcnNet, self).__init__()
        self.fi = math.floor(math.log2(si))
        self.MultiLevel_Spectral = MultiLevelSpectralBlock(64)
        self.cbam = TemporalSpatialBlock(657)
        self.global_weight = GlobalAdjLeaningLayer(64)

        self.gamma_x = GcnBlock(328, 128, dropout)
        self.gamma_x1 = GcnBlock(128, 16, dropout)
        self.local_gamma_x = GcnBlock(328, 128, dropout)
        self.local_gamma_x1 = GcnBlock(128, 16, dropout)

        self.beta_x = GcnBlock(164, 64, dropout)
        self.beta_x1 = GcnBlock(64, 16, dropout)
        self.local_beta_x = GcnBlock(164, 64, dropout)
        self.local_beta_x1 = GcnBlock(64, 16, dropout)

        self.alpha_x = GcnBlock(82, 32, dropout)
        self.alpha_x1 = GcnBlock(32, 8, dropout)
        self.local_alpha_x = GcnBlock(82, 32, dropout)
        self.local_alpha_x1 = GcnBlock(32, 8, dropout)

        self.delta_x = GcnBlock(41, 16, dropout)
        self.delta_x1 = GcnBlock(16, 4, dropout)
        self.local_delta_x = GcnBlock(41, 16, dropout)
        self.local_delta_x1 = GcnBlock(16, 4, dropout)

        self.theta_x = GcnBlock(41, 16, dropout)
        self.theta_x1 = GcnBlock(16, 4, dropout)
        self.local_theta_x = GcnBlock(41, 16, dropout)
        self.local_theta_x1 = GcnBlock(16, 4, dropout)

        self.calssifier = Classification()
        self.globalmeanpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, edges, local_graph_weight, global_mask):
        x = self.cbam(x.permute(0, 3, 2, 1))
        x = x.permute(0, 3, 2, 1)

        for i in range(1, self.fi - 2):
            if i <= self.fi - 7:
                if i == 1:
                    outputs, _ = self.MultiLevel_Spectral(x)
                else:
                    outputs, _ = self.MultiLevel_Spectral(outputs)
            elif i == self.fi - 6:
                if self.fi >= 8:
                    outputs, gamma = self.MultiLevel_Spectral(outputs)
                else:
                    outputs, gamma = self.MultiLevel_Spectral(x)
            elif i == self.fi - 5:
                outputs, beta = self.MultiLevel_Spectral(outputs)
            elif i == self.fi - 4:
                outputs, alpha = self.MultiLevel_Spectral(outputs)
            elif i == self.fi - 3:
                delta, theta = self.MultiLevel_Spectral(outputs)

        global_weight = self.global_weight(global_mask)
        # local_gamma_fea = self.local_gamma_x(torch.squeeze(gamma), edges, local_graph_weight)
        # local_gamma_fea = self.local_gamma_x1(local_gamma_fea, edges, local_graph_weight)
        # gamma_fea = self.gamma_x(torch.squeeze(gamma), edges, global_weight)
        # gamma_fea = self.gamma_x1(gamma_fea, edges, global_weight)

        # local_beta_fea = self.local_beta_x(torch.squeeze(beta), edges, local_graph_weight)
        # local_beta_fea = self.local_beta_x1(local_beta_fea, edges, local_graph_weight)
        # beta_fea = self.beta_x(torch.squeeze(beta), edges, global_weight)
        # beta_fea = self.beta_x1(beta_fea, edges, global_weight)

        # local_alpha_fea = self.local_alpha_x(torch.squeeze(alpha), edges, local_graph_weight)
        # local_alpha_fea = self.local_alpha_x1(local_alpha_fea, edges, local_graph_weight)
        # alpha_fea = self.alpha_x(torch.squeeze(alpha), edges, global_weight)
        # alpha_fea = self.alpha_x1(alpha_fea, edges, global_weight)

        local_delta_fea = self.local_delta_x(torch.squeeze(delta), edges, local_graph_weight)
        local_delta_fea = self.local_delta_x1(local_delta_fea, edges, local_graph_weight)
        delta_fea = self.delta_x(torch.squeeze(delta), edges, global_weight)
        delta_fea = self.delta_x1(delta_fea, edges, global_weight)

        # local_theta_fea = self.local_theta_x(torch.squeeze(theta), edges, local_graph_weight)
        # local_theta_fea = self.local_theta_x1(local_theta_fea, edges, local_graph_weight)
        # theta_fea = self.theta_x(torch.squeeze(theta), edges, global_weight)
        # theta_fea = self.theta_x1(theta_fea, edges, global_weight)

        # x = torch.cat((self.globalmeanpool(local_gamma_fea),self.globalmeanpool(gamma_fea)),dim=2)
        # x = torch.cat((self.globalmeanpool(local_beta_fea),self.globalmeanpool(beta_fea)),dim=2)
        # x = torch.cat((self.globalmeanpool(local_alpha_fea),self.globalmeanpool(alpha_fea)),dim=2)
        x = torch.cat((self.globalmeanpool(local_delta_fea),self.globalmeanpool(delta_fea)),dim=2)
        # x = torch.cat((self.globalmeanpool(local_theta_fea),self.globalmeanpool(theta_fea)),dim=2)

        out = self.calssifier(x)

        return out
