import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from models.Self_attention import Self_Attn


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal', gain=0.02):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 10 ** (opt.lr_first-((abs(opt.lr_last)-abs(opt.lr_first)) * epoch /(opt.niter-1)))
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(input_nc, output_nc, sym_nc, ngf, which_model_netG, norm='batch', init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'originalUnet':
        netG = UnetGenerator(input_nc, output_nc, ngf, gpu_ids=gpu_ids)
    elif which_model_netG == 'polyphaseUnet':
        netG = PolyUnetGenerator(input_nc, output_nc, ngf, gpu_ids=gpu_ids)
    elif which_model_netG == 'originalUnet_CCAM':
        netG = UnetGeneratorCCAM(input_nc, output_nc, sym_nc, ngf, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    # return netG
    return init_net(netG, init_type, 0.02, gpu_ids)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Classes
##############################################################################
class Cblock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(Cblock, self).__init__()
        self.block = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        return self.block(x)

class CRblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CRblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(True)) #

    def forward(self, x):
        return self.block(x)

class CBRblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CBRblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(True)) #

    def forward(self, x):
        return self.block(x)

class CSblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CSblock, self).__init__()
        self.block = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)
        self.activ = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block(x)
        return self.activ(x)

class encblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encblock, self).__init__()
        self.downconv = nn.MaxPool2d(2)
        # self.downconv = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0, bias=True)
        self.block1 = CBRblock(in_ch, out_ch)
        self.block2 = CBRblock(out_ch, out_ch)

    def forward(self, x):
        y = self.downconv(x)
        y1 = self.block1(y)
        return self.block2(y1)

class decblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decblock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, padding=0, stride=2, output_padding=0)
        self.block1 = CBRblock(in_ch, out_ch)
        self.block2 = CBRblock(out_ch, out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat((x2, upconved), dim=1)
        y = self.block1(x)
        y = self.block2(y)
        return y

class outblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outblock, self).__init__()
        self.block = Cblock(in_ch, out_ch, kernel_size=1, padding=0)
        # self.activ = nn.Softmax2d()
        self.activ = nn.Sigmoid()
    def forward(self, x):
        y = self.block(x)
        return y

################################################################################
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = UnetBlock1(input_nc, output_nc, ngf)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class UnetBlock1(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UnetBlock1, self).__init__()
        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)

        self.encoder1 = encblock(ngf, ngf * 2)
        self.encoder2 = encblock(ngf * 2, ngf * 4)
        self.encoder3 = encblock(ngf * 4, ngf * 8)
        self.encoder4 = encblock(ngf * 8, ngf * 16)
        self.decoder1 = decblock(ngf * 16, ngf * 8)
        self.decoder2 = decblock(ngf * 8, ngf * 4)
        self.decoder3 = decblock(ngf * 4, ngf * 2)
        self.decoder4 = decblock(ngf * 2, ngf)

        self.outblock = outblock(ngf, output_nc)

    def forward(self, input):
        x = self.inblock1(input)
        x = self.inblock2(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        y = self.decoder1(x4, x3)
        y = self.decoder2(y, x2)
        y = self.decoder3(y, x1)
        y = self.decoder4(y, x)
        y = self.outblock(y)
        return y + input

class UnetGeneratorCCAM(nn.Module):
    def __init__(self, input_nc, output_nc, sym_nc, ngf=64, gpu_ids=[]):
        super(UnetGeneratorCCAM, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = UnetBlock1CCAM(input_nc, output_nc, sym_nc, ngf)

    def forward(self, input, sym):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, sym, self.gpu_ids)
        # else:
        return self.model(input, sym)

class UnetBlock1CCAM(nn.Module):
    def __init__(self, input_nc, output_nc, sym_nc, ngf=64):
        super(UnetBlock1CCAM, self).__init__()

        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)
        self.ccamblock1 = CCAMBlock(ngf, 64, sym_nc)
        self.encoder1 = encblock(ngf, ngf * 2)
        self.ccamblock2 = CCAMBlock(ngf*2, 32, sym_nc)
        self.encoder2 = encblock(ngf * 2, ngf * 4)
        self.ccamblock3 = CCAMBlock(ngf*4, 16, sym_nc)
        self.encoder3 = encblock(ngf * 4, ngf * 8)
        self.ccamblock4 = CCAMBlock(ngf*8, 8, sym_nc)
        self.encoder4 = encblock(ngf * 8, ngf * 16)
        self.ccamblock5 = CCAMBlock(ngf*16, 4, sym_nc)
        self.decoder1 = decblock(ngf * 16, ngf * 8)
        self.ccamblock6 = CCAMBlock(ngf*8, 8, sym_nc)
        self.decoder2 = decblock(ngf * 8, ngf * 4)
        self.ccamblock7 = CCAMBlock(ngf*4, 16, sym_nc)
        self.decoder3 = decblock(ngf * 4, ngf * 2)
        self.ccamblock8 = CCAMBlock(ngf*2, 32, sym_nc)
        self.decoder4 = decblock(ngf * 2, ngf)
        self.outblock = outblock(ngf, output_nc)

    def forward(self, input, sym):
        x = self.inblock1(input)
        x = self.inblock2(x)
        x = self.ccamblock1(x, sym)
        x1 = self.encoder1(x)
        x1 = self.ccamblock2(x1, sym)
        x2 = self.encoder2(x1)
        x2 = self.ccamblock3(x2, sym)
        x3 = self.encoder3(x2)
        x3 = self.ccamblock4(x3, sym)
        x4 = self.encoder4(x3)
        x4 = self.ccamblock5(x4, sym)
        y = self.decoder1(x4, x3)
        y = self.ccamblock6(y, sym)
        y = self.decoder2(y, x2)
        y = self.ccamblock7(y, sym)
        y = self.decoder3(y, x1)
        y = self.ccamblock8(y, sym)
        y = self.decoder4(y, x)
        y = self.outblock(y)
        return y + input
########################################################################################################################

class subpixelPool(nn.Module):
    def __init__(self, input_nc):
        super(subpixelPool, self).__init__()
        self.input_nc = input_nc
        self.output_nc = input_nc*4

    def forward(self, input):
        output1 = input[:, :, ::2, ::2]
        output2 = input[:, :, ::2, 1::2]
        output3 = input[:, :, 1::2, ::2]
        output4 = input[:, :, 1::2, 1::2]
        return torch.cat([output1, output2, output3, output4], dim=1)


class unSubpixelPool(nn.Module):
    def __init__(self, input_nc):
        super(unSubpixelPool, self).__init__()
        self.input_nc = input_nc*2
        self.output_nc = int(input_nc/2)

    def forward(self, input):
        output = Variable(torch.cuda.FloatTensor(input.shape[0], self.output_nc, input.shape[2]*2, input.shape[3]*2)).zero_()
        output[:, :, ::2, ::2] = input[:, :int(self.input_nc/4), :, :]
        output[:, :, ::2, 1::2] = input[:, int(self.input_nc/4):int(self.input_nc/2), :, :]
        output[:, :, 1::2, ::2] = input[:, int(self.input_nc/2):int(self.input_nc*3/4), :, :]
        output[:, :, 1::2, 1::2] = input[:, int(self.input_nc*3/4):, :, :]
        return output

class polyencblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polyencblock, self).__init__()
        self.downconv = subpixelPool(in_ch)
        self.block1 = CBRblock(int(in_ch*4), out_ch)
        self.block2 = CBRblock(out_ch, out_ch)

    def forward(self, x):
        y = self.downconv(x)
        y1 = self.block1(y)
        return self.block2(y1)


class polyencSAblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polyencSAblock, self).__init__()
        self.downconv = subpixelPool(in_ch)
        self.block1 = CBRblock(int(in_ch*4), out_ch)
        self.block2 = CBRblock(out_ch, out_ch)
        self.attn = Self_Attn(out_ch)

    def forward(self, x):
        y = self.downconv(x)
        y1 = self.block1(y)
        y2 = self.block2(y1)
        return self.attn(y2)

class polyinencblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polyinencblock, self).__init__()
        self.downconv = subpixelPool(in_ch)
        self.block1 = CBRblock(int(in_ch*4), out_ch)
        self.block2 = CBRblock(out_ch, out_ch*2)

    def forward(self, x):
        y = self.downconv(x)
        y1 = self.block1(y)
        return self.block2(y1)


class polyinencSAblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polyinencSAblock, self).__init__()
        self.downconv = subpixelPool(in_ch)
        self.block1 = CBRblock(int(in_ch*4), out_ch)
        self.block2 = CBRblock(out_ch, out_ch*2)
        self.attn = Self_Attn(out_ch*2)

    def forward(self, x):
        y = self.downconv(x)
        y1 = self.block1(y)
        y2 = self.block2(y1)
        return self.attn(y2)

class polydecblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polydecblock, self).__init__()
        self.upconv = unSubpixelPool(in_ch)
        self.block1 = CBRblock(in_ch, out_ch)
        self.block2 = CBRblock(out_ch, out_ch*2)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat((x2, upconved), dim=1)
        y = self.block1(x)
        y = self.block2(y)
        return y


class polydecSAblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polydecSAblock, self).__init__()
        self.upconv = unSubpixelPool(in_ch)
        self.block1 = CBRblock(in_ch, out_ch)
        self.block2 = CBRblock(out_ch, out_ch*2)
        self.attn = Self_Attn(out_ch*2)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat((x2, upconved), dim=1)
        y = self.block1(x)
        y = self.block2(y)
        y = self.attn(y)
        return y

class polyoutdecblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polyoutdecblock, self).__init__()
        self.upconv = unSubpixelPool(in_ch)
        self.block1 = CBRblock(in_ch, out_ch)
        self.block2 = CBRblock(out_ch, out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat((x2, upconved), dim=1)
        y = self.block1(x)
        y = self.block2(y)
        return y


class polyoutdecSAblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(polyoutdecSAblock, self).__init__()
        self.upconv = unSubpixelPool(in_ch)
        self.block1 = CBRblock(in_ch, out_ch)
        self.block2 = CBRblock(out_ch, out_ch)
        self.attn = Self_Attn(out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat((x2, upconved), dim=1)
        y = self.block1(x)
        y = self.block2(y)
        y = self.attn(y)
        return y

################################################################################
class PolyUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, gpu_ids=[]):
        super(PolyUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = PolyUnetBlock(input_nc, output_nc, ngf)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class PolyUnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_residual=True):
        super(PolyUnetBlock, self).__init__()
        self.use_residual = use_residual
        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)

        self.encoder1 = polyencblock(ngf, ngf * 2)
        self.encoder2 = polyencblock(ngf * 2, ngf * 4)
        self.encoder3 = polyencblock(ngf * 4, ngf * 8)
        self.encoder4 = polyinencblock(ngf * 8, ngf * 16)
        self.decoder1 = polydecblock(ngf * 16, ngf * 8)
        self.decoder2 = polydecblock(ngf * 8, ngf * 4)
        self.decoder3 = polydecblock(ngf * 4, ngf * 2)
        self.decoder4 = polyoutdecblock(ngf * 2, ngf)

        self.outblock = outblock(ngf, output_nc)

    def forward(self, input):
        x = self.inblock1(input)
        x = self.inblock2(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        y = self.decoder1(x4, x3)
        y = self.decoder2(y, x2)
        y = self.decoder3(y, x1)
        y = self.decoder4(y, x)
        y = self.outblock(y)

        if self.use_residual:
            y = y + input

        return y


class PolySAUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_residual=True):
        super(PolySAUnet, self).__init__()
        self.use_residual = use_residual
        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)

        self.encoder1 = polyencSAblock(ngf, ngf * 2)
        self.encoder2 = polyencSAblock(ngf * 2, ngf * 4)
        self.encoder3 = polyencSAblock(ngf * 4, ngf * 8)
        self.encoder4 = polyinencSAblock(ngf * 8, ngf * 16)
        self.decoder1 = polydecblock(ngf * 16, ngf * 8)
        self.decoder2 = polydecblock(ngf * 8, ngf * 4)
        self.decoder3 = polydecblock(ngf * 4, ngf * 2)
        self.decoder4 = polyoutdecblock(ngf * 2, ngf)

        self.outblock = outblock(ngf, output_nc)

    def forward(self, input):
        x = self.inblock1(input)
        x = self.inblock2(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        y = self.decoder1(x4, x3)
        y = self.decoder2(y, x2)
        y = self.decoder3(y, x1)
        y = self.decoder4(y, x)
        y = self.outblock(y)

        if self.use_residual:
            y = y + input

        return y


class PolySAUnet_v2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)

        self.encoder1 = polyencblock(ngf, ngf * 2)
        self.encoder2 = polyencblock(ngf * 2, ngf * 4)
        self.encoder3 = polyencblock(ngf * 4, ngf * 8)
        self.encoder4 = polyinencblock(ngf * 8, ngf * 16)
        self.decoder1 = polydecSAblock(ngf * 16, ngf * 8)
        self.decoder2 = polydecSAblock(ngf * 8, ngf * 4)
        self.decoder3 = polydecSAblock(ngf * 4, ngf * 2)
        self.decoder4 = polyoutdecSAblock(ngf * 2, ngf)

        self.outblock = outblock(ngf, output_nc)

    def forward(self, input):
        x = self.inblock1(input)
        x = self.inblock2(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        y = self.decoder1(x4, x3)
        y = self.decoder2(y, x2)
        y = self.decoder3(y, x1)
        y = self.decoder4(y, x)
        y = self.outblock(y)

        if self.use_residual:
            y = y + input

        return y


class PolySAUnet_merge(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(PolySAUnet_merge, self).__init__()
        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)

        self.encoder1 = polyencSAblock(ngf, ngf * 2)
        self.encoder2 = polyencSAblock(ngf * 2, ngf * 4)
        self.encoder3 = polyencSAblock(ngf * 4, ngf * 8)
        self.encoder4 = polyinencSAblock(ngf * 8, ngf * 16)
        self.decoder1 = polydecblock(ngf * 16, ngf * 8)
        self.decoder2 = polydecblock(ngf * 8, ngf * 4)
        self.decoder3 = polydecblock(ngf * 4, ngf * 2)
        self.decoder4 = polyoutdecblock(ngf * 2, ngf)

        self.outblock = outblock(ngf, output_nc)
        self.adaptive_residual_conv = nn.Conv2d(output_nc * 2, output_nc, kernel_size=1)

    def forward(self, input):
        x = self.inblock1(input)
        x = self.inblock2(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        y = self.decoder1(x4, x3)
        y = self.decoder2(y, x2)
        y = self.decoder3(y, x1)
        y = self.decoder4(y, x)
        y = self.outblock(y)

        res = y + input

        res_non_res = torch.cat((y, res), dim=1)
        out = self.adaptive_residual_conv(res_non_res)

        return out


class PolySAUnet_merge_v2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super().__init__()
        self.inblock1 = CBRblock(input_nc, ngf)
        self.inblock2 = CBRblock(ngf, ngf)

        self.encoder1 = polyencblock(ngf, ngf * 2)
        self.encoder2 = polyencblock(ngf * 2, ngf * 4)
        self.encoder3 = polyencblock(ngf * 4, ngf * 8)
        self.encoder4 = polyinencblock(ngf * 8, ngf * 16)
        self.decoder1 = polydecSAblock(ngf * 16, ngf * 8)
        self.decoder2 = polydecSAblock(ngf * 8, ngf * 4)
        self.decoder3 = polydecSAblock(ngf * 4, ngf * 2)
        self.decoder4 = polyoutdecSAblock(ngf * 2, ngf)

        self.outblock = outblock(ngf, output_nc)
        self.adaptive_residual_conv = nn.Conv2d(output_nc * 2, output_nc, kernel_size=1)

    def forward(self, input):
        x = self.inblock1(input)
        x = self.inblock2(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        y = self.decoder1(x4, x3)
        y = self.decoder2(y, x2)
        y = self.decoder3(y, x1)
        y = self.decoder4(y, x)
        y = self.outblock(y)

        res = y + input

        res_non_res = torch.cat((y, res), dim=1)
        out = self.adaptive_residual_conv(res_non_res)

        return out


class Wnet_PolySA(nn.Module):

    def __init__(self, in_chans, out_chans, chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        ch = chans

        self.Unet1 = PolySAUnet_merge(in_chans, out_chans, ch)
        self.Unet2 = PolySAUnet(in_chans, out_chans, ch, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output


class Wnet_PolySA_v2(nn.Module):

    def __init__(self, in_chans, out_chans, chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        ch = chans

        self.Unet1 = PolySAUnet_merge_v2(in_chans, out_chans, ch)
        self.Unet2 = PolySAUnet_v2(in_chans, out_chans, ch, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output

################################################################################

class CCAMBlock(nn.Module):
    def __init__(self, input_nc, pool_size, sym_nc):
        super(CCAMBlock, self).__init__()
        self.avgpool = nn.AvgPool2d(pool_size)
        self.MLPe = nn.Linear(sym_nc, input_nc)
        self.MLPm1 = nn.Linear(input_nc * 2, input_nc/2)
        self.MLPm2 = nn.Linear(input_nc / 2, input_nc)
        self.activ = nn.Sigmoid()
    def forward(self, input, sym):
        x = self.avgpool(input)
        x = x.view(input.size(0),-1)

        y = self.MLPe(sym)
        y = torch.cat((x, y), 1)

        y = self.MLPm1(y)
        y = self.MLPm2(y)
        y = self.activ(y)
        y = y.view(y.size(0), -1, 1, 1)
        y = torch.mul(input, y)

        return y
