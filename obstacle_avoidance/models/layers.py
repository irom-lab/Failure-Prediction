import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def KLDiv_gaussian(mu1, var1, mu2, var2, var_is_logvar=True):
    if var_is_logvar:
        var1 = torch.exp(var1)
        var2 = torch.exp(var2)

    mu1 = torch.flatten(mu1)  # make sure we are 1xd so torch functions work as expected
    var1 = torch.flatten(var1)
    mu2 = torch.flatten(mu2)
    var2 = torch.flatten(var2)

    kl_div = 1/2 * torch.log(torch.div(var2, var1))
    kl_div += 1/2 * torch.div(var1 + torch.pow(mu2 - mu1, 2), var2)
    kl_div -= 1/2  # one for each dimension

    return torch.sum(kl_div)


class StochasticLayer(nn.Module):
    def __init__(self, weights_size, bias=True):
        super().__init__()
        self.weights_size = weights_size
        self.bias = bias

        self.mu = nn.Parameter(torch.ones(weights_size))
        self.logvar = nn.Parameter(torch.zeros(weights_size))
        self.b_mu = nn.Parameter(torch.zeros(weights_size[0])) if bias else None
        self.b_logvar = nn.Parameter(torch.zeros(weights_size[0])) if bias else None

        self.init_mu()
        self.init_logvar()

        self.stdev_xi = None
        self.b_stdev_xi = None

    def init_mu(self):
        n = self.mu.size(1)
        stdev = math.sqrt(1./n)
        self.mu.data.uniform_(-stdev, stdev)
        if self.bias:
            self.b_mu.data.uniform_(-stdev, stdev)

    def init_logvar(self, logvar=0., b_logvar=0.):
        self.logvar.data.zero_()
        self.logvar.data += logvar
        if self.bias:
            self.b_logvar.data.zero_()
            self.b_logvar.data += b_logvar

    def init_xi(self):
        stdev = torch.exp(0.5 * self.logvar)
        xi = stdev.data.new(stdev.size()).normal_(0, 1)
        self.stdev_xi = stdev * xi
        if self.bias:
            b_stdev = torch.exp(0.5 * self.b_logvar)
            b_xi = b_stdev.data.new(b_stdev.size()).normal_(0, 1)
            self.b_stdev_xi = b_stdev * b_xi

    def forward(self, x):
        assert self.stdev_xi is not None
        layer = self.mu + self.stdev_xi
        b_layer = self.b_mu + self.b_stdev_xi if self.bias else None
        out = self.operation(x, layer, b_layer)
        return out

    def operation(self, x, weight, bias):
        raise NotImplementedError

    def to_str(self):
        print("mu", self.mu.data.flatten()[:5].to('cpu').numpy())

    def calc_kl_div(self, prior):
        mu1 = self.mu
        logvar1 = self.logvar
        mu2 = prior.mu.clone().detach()
        logvar2 = prior.logvar.clone().detach()
        kl_div = KLDiv_gaussian(mu1, logvar1, mu2, logvar2, var_is_logvar=True)

        if self.bias:
            b_mu1 = self.b_mu
            b_logvar1 = self.b_logvar
            b_mu2 = prior.b_mu.clone().detach()
            b_logvar2 = prior.b_logvar.clone().detach()
            kl_div += KLDiv_gaussian(b_mu1, b_logvar1, b_mu2, b_logvar2, var_is_logvar=True)

        return kl_div


class StochasticLinear(StochasticLayer):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__((output_dim, input_dim), bias=bias)

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)


class NotStochasticLinear(StochasticLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_xi()
        self.init_logvar(-float('Inf'), -float('Inf'))

    def init_xi(self):
        self.stdev_xi = 0
        self.b_stdev_xi = 0


class StochasticConv2d(StochasticLayer):
    # def __init__(self, weights_size, stride, padding, bias=True):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        # super().__init__( ,bias=bias)
        # 1, n_filt, kernel_size = (4, 4), stride = (3, 3), padding = 0),
        # (n_filt, 1, 4, 4), 3, 0
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        else:
            assert type(kernel_size) == tuple

        weights_size = (out_channels, in_channels, *kernel_size)
        super().__init__(weights_size, bias=bias)
        self.stride = stride
        self.padding = padding

    def _init_mu(self):
        torch.nn.init.kaiming_normal_(self.mu)

    def operation(self, x, weight, bias):
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


class NotStochasticConv2d(StochasticConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_xi()
        self.init_logvar(-float('Inf'), -float('Inf'))

    def init_xi(self):
        self.stdev_xi = 0
        self.b_stdev_xi = 0


class StochasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.compatible_classes = (StochasticLayer,
                                   StochasticLinear,
                                   StochasticConv2d,
                                   )

    def forward(self, x):
        raise NotImplementedError()

    def init_xi(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.init_xi(*args, **kwargs)
            elif layer.__class__ == nn.Sequential:
                pass  # call init_xi on everything inside the sequential layer

    def to_str(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.to_str(*args, **kwargs)

    def init_logvar(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.init_logvar(*args, **kwargs)

    def init_mu(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.init_mu(*args, **kwargs)

    def project_logvar(self, prior, a=2):
        for (name, layer), (prior_name, prior_layer) in zip(self.named_modules(), prior.named_modules()):
            if layer.__class__ in self.compatible_classes:
                layer.project_logvar(prior_layer, a=a)

    def calc_kl_div(self, prior, device=None):
        if device is not None:
            kl_div = torch.tensor(0., dtype=torch.float).to(device)
        else:
            kl_div = torch.tensor(0., dtype=torch.float)

        for (name, layer), (prior_name, prior_layer) in zip(self.named_modules(), prior.named_modules()):
            if layer.__class__ in self.compatible_classes:
                kl_div += layer.calc_kl_div(prior_layer)

        return kl_div
