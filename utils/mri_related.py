import torch
from utils import WarpForward, WarpAdjoint


def fft2(x, dim=(-2,-1)):
    return torch.fft.fft2(x, dim=dim, norm='ortho')


def ifft2(X, dim=(-2,-1)):
    return torch.fft.ifft2(X, dim=dim, norm='ortho')


def fft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(x, dim), dim), dim)


def ifft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(x, dim), dim), dim)


class MulticoilForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = torch.unsqueeze(image[:,0], self.coil_axis) * smaps
        else:
            coilimg = torch.unsqueeze(image, self.coil_axis) * smaps
        kspace = self.fft2(coilimg)
        masked_kspace = kspace * mask
        return masked_kspace


class MulticoilAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, kspace, mask, smaps):
        masked_kspace = kspace * mask
        coilimg = self.ifft2(masked_kspace)
        img = torch.sum(torch.conj(smaps) * coilimg, self.coil_axis)

        if self.channel_dim_defined:
            return torch.unsqueeze(img, 1)
        else:
            return img


class ForwardOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2

    def forward(self, image, mask):
        kspace = self.fft2(image)
        masked_kspace = kspace * mask
        return masked_kspace


class MulticoilMotionForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.W = WarpForward()
        self.A = MulticoilForwardOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.channel_dim_defined = channel_dim_defined

    def forward(self, x, mask, smaps, u):
        if self.channel_dim_defined:
            x = self.W(x[:,0], u)
        else:
            x = self.W(x, u)
        y = self.A(x, mask, smaps)
        return y


class MulticoilMotionAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.AH = MulticoilAdjointOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.WH = WarpAdjoint()
        self.channel_dim_defined = channel_dim_defined

    def forward(self, y, mask, smaps, u):
        x = self.AH(y, mask, smaps)
        x = self.WH(x, u)
        if self.channel_dim_defined:
            return torch.unsqueeze(x, 1)
        else:
            return x


class AdjointOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2

    def forward(self, kspace, mask):
        masked_kspace = kspace * mask
        img = self.ifft2(masked_kspace)
        return img

