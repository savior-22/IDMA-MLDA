import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.autograd import gradcheck


class GMMLoss(nn.Module):
    def __init__(self, num_classes, zdim, reduction='mean'):
        super(GMMLoss, self).__init__()
        self.reduction = reduction      # 损失函数输出值的缩减方式
        self.zdim = zdim
        self.num_classes = num_classes
        self.theta_p = nn.Parameter(torch.tensor([4.5, 1/5, 1/5]))          #4.5, 1/5, 1/5 gcngat2
        # self.theta_p = nn.Parameter((torch.full((self.num_classes,), 1.0 / self.num_classes)))
        self.u_p = nn.Parameter(torch.zeros(self.num_classes, self.zdim))
        self.lambda_p = nn.Parameter(torch.ones(self.num_classes, self.zdim))
        # self.register_buffer('lambda_p', torch.ones(self.num_classes, self.zdim))

    def gradient_check(self, mean, logvar, z_tilde):
        # 示例输入
        batch_size = mean.size(0)
        input_args = (mean, logvar, z_tilde)

        # 运行梯度检查
        print("Running gradient check...")
        gradcheck(self.forward, input_args, eps=1e-6, atol=1e-4)
        print("Gradient check passed.")

    def forward(self, mean, logvar, z_tilde):
        GMM_mean = self.u_p
        GMM_var = self.lambda_p       # dim=[1,3]
        weight = self.theta_p           # [[1/3],[1/3],[1/3]]

        z_mean_t = torch.unsqueeze(mean, dim=1).expand(-1, self.num_classes, -1)  # .transpose(2,1)
        z_log_var_t = torch.unsqueeze(logvar, dim=1).expand(-1, self.num_classes, -1)  # .transpose(2,1)
        Z = (torch.unsqueeze(z_tilde, dim=1).expand(-1, self.num_classes, -1))  # .transpose(2,1)

        # Z = tf.transpose(tf.tile(tf.expand_dims(z_tilde, dim=1), [1, self.num_classes, 1]), [ 2, 1])
        # print('self.mean',self.enc_mean)
        a = torch.unsqueeze(torch.log(weight), dim=1) - 0.5 * torch.log(2 * math.pi * GMM_var)
        b = (Z - GMM_mean) ** 2
        c = (2 * GMM_var)
        # print('gmmsize()', weight.size(), GMM_var.size(), z_log_var_t.size())
        p_c_z = torch.exp(torch.sum((a - b / c), dim=2)) + 1e-10  # 应该是bs*nclasses, dim应该要改一下

        gamma = p_c_z / torch.sum(p_c_z, dim=-1, keepdim=True)
        gamma_t = torch.unsqueeze(gamma, dim=2).expand(-1, -1, self.zdim)

        g = torch.exp(z_log_var_t) / GMM_var
        h = (z_mean_t - GMM_mean) ** 2 / GMM_var
        d = torch.log(GMM_var) + g + h  # +f

        loss = torch.sum(0.5 * gamma_t * d, dim=(1, 2))

        loss = loss - 0.5 * torch.sum(logvar + 1, dim=-1)

        loss = loss - torch.sum(torch.log(weight) * gamma, dim=-1) + torch.sum(torch.log(gamma) * gamma, dim=-1)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction != 'none':
            raise NotImplementedError
        # print('gmm-loss', loss)
        return loss


