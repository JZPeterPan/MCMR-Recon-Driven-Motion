import torch
from splatting import splatting_function


def warp_torch(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    # flo = torch.flip(flo, dims=[1])
    # vgrid = Variable(grid) + flo
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()

    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    # return output * mask
    return output


class WarpForward(torch.nn.Module):
    def forward(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = u.shape[:-1]
        M, N = u.shape[-3:-1]
        x = torch.repeat_interleave(torch.unsqueeze(x, -3), repeats=u.shape[-4], dim=-3)
        x = torch.reshape(x, (-1, 1, M, N)) # [batch, frames * frames_all, 1, M, N]
        u = torch.reshape(u, (-1, M, N, 2)) # [batch, frames * frames_all, M, N, 2]
        x_re = torch.real(x).contiguous()
        x_im = torch.imag(x).contiguous()
        out_re = warp_torch(x_re, u.permute(0, 3, 1, 2))
        out_im = warp_torch(x_im, u.permute(0, 3, 1, 2))
        Wx = torch.complex(out_re, out_im)

        return torch.reshape(Wx, out_shape)


class WarpAdjoint(torch.nn.Module):
    def forward(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, frames_all, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = u.shape[:-1]
        M, N = u.shape[-3:-1]
        x = torch.reshape(x, (-1, 1, M, N)) # [batch * frames * frames_all, 1, M, N]
        u = torch.reshape(u, (-1, M, N, 2)) # [batch * frames * frames_all, M, N, 2]
        x_re = torch.real(x).contiguous()
        x_im = torch.imag(x).contiguous()
        out_re = splatting_function("summation", x_re, u.permute(0, 3, 1, 2))
        out_im = splatting_function("summation", x_im, u.permute(0, 3, 1, 2))
        x_warpT = torch.complex(out_re, out_im)
        x_warpT = torch.reshape(x_warpT, out_shape)
        x_warpT = torch.sum(x_warpT, -3)
        return x_warpT
