import torch
from torch import nn
from torch.nn import functional as F
from data.data_transforms import kspace_to_nchw, nchw_to_kspace
from models.attention import ChannelAttention
from models.CBAM import CBAM
from models.Self_attention import Self_Attn
from models.my_CS_attention import CBSA
from models.polyphase_Unet import subpixelPool, unSubpixelPool, PolyUnetBlock

class ConvReLU(nn.Module):
    """
    A Simple Conv-ReLU function without any normalization
    """
    def __init__(self, in_chans, out_chans, stride=1):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(tensor)

    def __repr__(self):
        return f'ConvReLU(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(tensor)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ConvBlockCBSA(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),
        )

        self.attention = CBSA(out_chans)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        tensor = self.layers(tensor)
        out = self.attention(tensor)
        return out

    def __repr__(self):
        return f'ConvBlockCBSA(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ConvBlockCBAM(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2, reduction_ratio=16):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.reduction_ratio = reduction_ratio

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),
        )

        self.CBAM = CBAM(out_chans, reduction_ratio=reduction_ratio)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        out = self.layers(tensor)
        out = self.CBAM(out)

        return out

    def __repr__(self):
        return f'ConvBlockCBAM(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ConvBlockSA(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2, reduction_ratio=16):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.reduction_ratio = reduction_ratio

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            nn.LeakyReLU(),
        )

        self.SA = Self_Attn(out_chans)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        out = self.layers(tensor)
        out = self.SA(out)

        return out

    def __repr__(self):
        return f'ConvBlockSA(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ConvBlockBatch(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(),

            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(tensor)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class AttConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, stride=2):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(),
        )

        self.layers2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(),
        )

        self.ca1 = ChannelAttention(num_chans=out_chans, reduction=16, use_gap=True, use_gmp=False)
        self.ca2 = ChannelAttention(num_chans=out_chans, reduction=16, use_gap=True, use_gmp=False)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        out1 = self.layers1(tensor)
        out1 = self.ca1(out1)
        out2 = self.layers2(out1)
        out2 = self.ca2(out2)
        return out2

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class UnetPS(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.upPS = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = self.upPS(output)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class Unet_CBSA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockCBSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockCBSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockCBSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class UnetSA(nn.Module):

    '''
    Due to computational overhead, Self-attention is applied only in the middle layer,
    and after the skip connections of every level
    '''

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockSA(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class UnetSA_multichan(nn.Module):

    '''
    Due to computational overhead, Self-attention is applied only in the middle layer,
    and after the skip connections of every level
    '''

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockSA(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor[:, 60:120, :, :]

        return output


class UnetSA_v2(nn.Module):

    '''
    Due to computational overhead, Self-attention is applied only in the middle layer,
    and after the skip connections of every level
    '''

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockSA(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class UnetSA_bypass(nn.Module):

    '''
    Due to computational overhead, Self-attention is applied only in the middle layer,
    and after the skip connections of every level
    '''

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockSA(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlockSA(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class UnetSA_light(nn.Module):

    '''
    Due to computational overhead, Self-attention is applied only in the middle layer,
    and after the skip connections of every level
    '''

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class Unet_CBAM(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True, reduction_ratio=16):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual
        self.reduction_ratio = reduction_ratio

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockCBAM(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlockCBAM(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockCBAM(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockCBAM(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockCBAM(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor

        return output


class Unet_merge(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Unet_mergeCBSA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockCBSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockCBSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockCBSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Unet_mergeSA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Unet_mergeSA_multichan(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor[:, 60:120, :, :]

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Unet_mergeSA_multichan_MP(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Unet_mergeSA_v2(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockSA(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockSA(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockSA(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)
        self.final_attention = Self_Attn(out_chans)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)
        final_output = self.final_attention(final_output)

        return final_output


class Unet_mergeSA_bypass(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockSA(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlockSA(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Unet_merge_CBAM(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockCBAM(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlockCBAM(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockCBAM(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockCBAM(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockCBAM(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.merge_residual_conv = nn.Conv2d(out_chans * 2, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        non_residual_ouptut = output
        residual_output = output + tensor

        res_non_res = torch.cat((non_residual_ouptut, residual_output), dim=1)

        final_output = self.merge_residual_conv(res_non_res)

        return final_output


class Wnet_half_residual_v2(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_merge(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = Unet(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output


class Wnet_CBSA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeCBSA(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = Unet_CBSA(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output


class Wnet_SA_multichan(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeSA_multichan(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = UnetSA(out_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output


class Wnet_SA_multichan_MP(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeSA_multichan_MP(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = UnetSA(out_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output


class Wnet_CBSA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeCBSA(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = Unet_CBSA(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        return refine_output


class Wnet_CBAM(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_merge_CBAM(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = Unet_CBAM(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        if self.use_residual:
            refine_output = tensor + refine_output

        return refine_output


class Wnet_SA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeSA(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = UnetSA(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        if self.use_residual:
            refine_output = tensor + refine_output

        return refine_output


class Wnet_SA_v2(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeSA_v2(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = UnetSA_v2(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        if self.use_residual:
            refine_output = tensor + refine_output

        return refine_output


class Wnet_SA_bypass(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans

        self.Unet1 = Unet_mergeSA_bypass(in_chans, out_chans, ch, num_pool_layers)
        self.Unet2 = UnetSA_bypass(in_chans, out_chans, ch, num_pool_layers, use_residual=False)

    def forward(self, tensor):

        mid_output = self.Unet1(tensor)
        refine_output = self.Unet2(mid_output)

        if self.use_residual:
            refine_output = tensor + refine_output

        return refine_output


class DCN(nn.Module):

    def __init__(self, in_chans, out_chans, chans, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.use_residual = use_residual

        ch = chans
        self.CBR1 = ConvBlock(in_chans, ch, stride=1)
        self.CBR2 = ConvBlock(ch, ch*2, stride=1)
        self.CBR3 = ConvBlock(ch*2, ch*4, stride=1)
        self.CBR4 = ConvBlock(ch*4, ch*2, stride=1)
        self.CBR5 = ConvBlock(ch*2, ch, stride=1)

        self.conv_final = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):

        output = self.CBR1(tensor)
        output = self.CBR2(output)
        output = self.CBR3(output)
        output = self.CBR4(output)
        output = self.CBR5(output)

        output = self.conv_final(output)

        if self.use_residual:
            output = output + tensor

        return output


class DCN_light(nn.Module):

    def __init__(self, in_chans, out_chans, chans, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.use_residual = use_residual

        ch = chans
        self.CBR1 = ConvBlock(in_chans, ch*2, stride=1)
        self.CBR2 = ConvBlock(ch*2, ch*4, stride=1)
        self.CBR3 = ConvBlock(ch*4, ch*2, stride=1)

        self.conv_final = nn.Conv2d(ch*2, out_chans, kernel_size=1)

    def forward(self, tensor):

        output = self.CBR1(tensor)
        output = self.CBR2(output)
        output = self.CBR3(output)

        output = self.conv_final(output)

        if self.use_residual:
            output = output + tensor

        return output


class CNN(nn.Module):

    def __init__(self, in_chans, out_chans, chans, use_residual=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.use_residual = use_residual

        ch = chans
        self.conv1 = ConvReLU(in_chans, ch*2, stride=1)
        self.conv2 = ConvReLU(ch*2, ch*4, stride=1)
        self.conv3 = ConvReLU(ch*4, ch*2, stride=1)

        self.conv_final = nn.Conv2d(ch*2, out_chans, kernel_size=1)

    def forward(self, tensor):
        output = self.conv1(tensor)
        output = self.conv2(output)
        output = self.conv3(output)

        output = self.conv_final(output)

        if self.use_residual:
            output = output + tensor

        return output


class UnetCA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

        self.ca = ChannelAttention(num_chans=out_chans, reduction=2, use_gap=True, use_gmp=False)
        self.conv3 = nn.Conv2d(out_chans, out_chans // 2, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)

        if self.use_residual:
            mid_output = output + tensor

        output = self.ca(mid_output)
        final_output = self.conv3(output)

        return mid_output, final_output


class UnetBatch(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlockBatch(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlockBatch(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlockBatch(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockBatch(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlockBatch(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor


        return output


class AttUnet(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([AttConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [AttConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = AttConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [AttConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [AttConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)
        if self.use_residual:
            output = output + tensor
        return output


class DoubleUnet(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, use_residual=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual

        ch = chans
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, stride=1)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, stride=2)]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here
        self.conv = ConvBlock(ch, ch, stride=2)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, stride=1)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, stride=1)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)
        self.conv_final = nn.Conv2d(out_chans, 1, kernel_size=1)

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)
        semi_output = self.conv2(output)
        if self.use_residual:
            semi_output = semi_output + tensor
        rss_output = self.conv_final(semi_output)
        return semi_output, rss_output