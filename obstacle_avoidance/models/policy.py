import torch.nn as nn
import torch
from models.layers import StochasticLinear as SLinear
from models.layers import StochasticConv2d as SConv2d
from models.layers import NotStochasticLinear as Linear
from models.layers import NotStochasticConv2d as Conv2d
from models.layers import StochasticModel


class Policy(nn.Module):
    def __init__(self, output_size=7):
        super().__init__()
        # 50x50 input
        n_filt = 64
        self.bn = nn.BatchNorm2d(n_filt)
        self.act = nn.ReLU()
        self.conv = nn.Sequential(nn.Conv2d(1, n_filt, kernel_size=4, stride=3, padding=0),
                                  self.bn,
                                  self.act,
                                  nn.Conv2d(n_filt, n_filt, kernel_size=3, stride=2, padding=0),
                                  self.bn,
                                  self.act,
                                  nn.Conv2d(n_filt, n_filt, kernel_size=3, stride=1, padding=0),
                                  self.bn,
                                  self.act,
                                  nn.Conv2d(n_filt, n_filt, kernel_size=3, stride=1, padding=0),
                                  self.bn,
                                  self.act)
        self.fc_out = nn.Sequential(nn.Linear(n_filt, n_filt),
                                    self.act,
                                    nn.Linear(n_filt, output_size),
                                    nn.Softmax())

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        x = self.fc_out(x)
        return x


class FailureDetectorStack(StochasticModel):
    def __init__(self, linear=nn.Linear, conv=nn.Conv2d):
        super().__init__()
        n_filt = 64
        self.n_stacked = 4
        self.n_filt = n_filt
        self.output_size = 2
        self.act = nn.ReLU()
        self.conv = nn.Sequential(conv(self.n_stacked, n_filt, kernel_size=(4, 4), stride=(3, 3), padding=0),  # (50-4)//3+1=16
                                  self.act,
                                  conv(n_filt, n_filt, kernel_size=(3, 3), stride=(2, 2), padding=0),  # (16-3)//2+1=7
                                  self.act,
                                  conv(n_filt, n_filt, kernel_size=(3, 3), stride=(2, 2), padding=0),  # (7-3)/2+1=3
                                  self.act)
        self.fc_out = nn.Sequential(linear(9*n_filt, 3*n_filt),
                                    self.act,
                                    linear(3*n_filt, n_filt),
                                    self.act,
                                    linear(n_filt, self.output_size),
                                    nn.Softmax())

    def forward(self, x):
        num_imgs = x.shape[1]
        batch = x.shape[0]
        output = torch.zeros([batch, num_imgs, self.output_size], dtype=torch.float).to(x.device)

        start_img = torch.unsqueeze(x[:, 0], dim=1)
        for i in range(self.n_stacked - 1):
            x = torch.cat((start_img, x), dim=1)

        for i in range(num_imgs):
            curr = x[:,i]
            for j in range(1, self.n_stacked):
                curr = torch.cat((curr, x[:, i+j]), dim=1)

            curr = self.conv(curr)
            curr = curr.flatten(1)
            # curr = curr.mean(dim=[2, 3])
            output[:,i] = self.fc_out(curr)

        return output

    def next(self, x):
        # assuming x is correct input size, bx4x50x50
        x = self.conv(x)
        x = x.flatten(1)
        output = self.fc_out(x)
        return output


class NSFailureDetectorStack(FailureDetectorStack):
    def __init__(self):
        super().__init__(linear=Linear, conv=Conv2d)


class SFailureDetectorStack(FailureDetectorStack):
    def __init__(self):
        super().__init__(linear=SLinear, conv=SConv2d)


class FailureDetectorSimple(StochasticModel):
    def __init__(self, linear=nn.Linear, conv=nn.Conv2d):
        super().__init__()
        n_filt = 64
        self.n_filt = n_filt
        self.output_size = 2
        self.act = nn.ReLU()
        self.conv = nn.Sequential(conv(1, n_filt, kernel_size=(4, 4), stride=(3, 3), padding=0),  # (50-4)//3+1=16
                                  self.act,
                                  conv(n_filt, n_filt, kernel_size=(3, 3), stride=(2, 2), padding=0),  # (16-3)//2+1=7
                                  self.act,
                                  conv(n_filt, n_filt, kernel_size=(3, 3), stride=(2, 2), padding=0),  # (7-3)/2+1=3
                                  self.act)
        self.fc_out = nn.Sequential(linear(9*n_filt, 3*n_filt),
                                    self.act,
                                    linear(3*n_filt, n_filt),
                                    self.act,
                                    linear(n_filt, self.output_size),
                                    nn.Softmax())

    def forward(self, x):
        num_imgs = x.shape[1]
        batch = x.shape[0]
        output = torch.zeros([batch, num_imgs, self.output_size], dtype=torch.float).to(x.device)

        fixed_detection = torch.zeros([batch, self.output_size], dtype=torch.float).to(x.device)
        fixed_detection[:, 1] += 1

        for i in range(num_imgs):
            curr = self.conv(x[:, i])
            curr = curr.flatten(1)
            # curr = curr.mean(dim=[2, 3])
            output_temp = self.fc_out(curr)
            output[:, i] = output_temp

        return output


class NSFailureDetectorSimple(FailureDetectorSimple):
    def __init__(self):
        super().__init__(linear=Linear, conv=Conv2d)


class SFailureDetectorSimple(FailureDetectorSimple):
    def __init__(self):
        super().__init__(linear=SLinear, conv=SConv2d)
