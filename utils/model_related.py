import torch,numpy as np, torch.nn.functional as F
from torchvision.transforms import ColorJitter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad(inp, divisor=8):
    pad_x = int(np.ceil(inp.shape[-2] / divisor)) * divisor - inp.shape[-2]
    pad_y = int(np.ceil(inp.shape[-1] / divisor)) * divisor - inp.shape[-1]
    inp = torch.nn.functional.pad(inp, (pad_y, 0, pad_x, 0))
    return inp, {'pad_x': pad_x, 'pad_y': pad_y}


def brightness_augmentation_torch(volume):
    [_, f, h, w] = volume.size()
    volume = torch.reshape(volume, (1, 1, f * h, w)).repeat_interleave(3, dim=1)
    volume = ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0.5/3.14)(volume.type(torch.uint8)).type(torch.float32)
    volume = torch.reshape(volume[:, 0, ...], (1, f, h, w))
    return volume


def normalize_3d_torch(volume, scale=255):
    max_3d = torch.max(volume.abs())
    min_3d = torch.min(volume.abs())
    volume = (volume - min_3d)/(max_3d - min_3d)
    volume = scale * volume
    return volume


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def unpad(inp, pad_x, pad_y):
    return inp[..., pad_x:, pad_y:]


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow, factor, mode='bilinear'):
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    return factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def switch2frameblock(img):
    f_pre = torch.cat((img[:, -1:, ...], img[:, :-1, ...]), dim=1)
    f_post = torch.cat((img[:, 1:, ...], img[:, :1, ...]), dim=1)
    img = torch.cat((img, f_pre, f_post), dim=2)
    return img


def neighboring_frame_select(input, slc, neighboring_frame, frame_dim=1):
    """
    the input is regarded as cyclic.
    :param input:
    :param slc:
    :param neighboring_frame:
    :param frame_dim:
    :return:
    """
    nfr = input.shape[frame_dim]
    if isinstance(neighboring_frame, int): assert 2*neighboring_frame+1 <= nfr
    # alternative: for neighboring_frame == 'all' we can also shift nothing
    shift_offset = int(nfr/2) - slc if neighboring_frame == 'all' else neighboring_frame - slc

    input_shifted = torch.roll(input, shift_offset, dims=frame_dim)
    output = torch.swapaxes(input_shifted, frame_dim, 0)
    if isinstance(neighboring_frame, int):
        output = output[:2*neighboring_frame+1, ...]
    output = torch.swapaxes(output, 0, frame_dim)
    return output


def group_mode_img_select_torch(volume, slc, neighbor_fr=4, abs=True):
    nfr = volume.shape[1]
    assert volume.dim() == 4
    if isinstance(neighbor_fr, int): assert 2*neighbor_fr+1 <= nfr

    if abs:
        volume = torch.abs(volume)

    img1 = neighboring_frame_select(volume, slc, neighbor_fr, frame_dim=1)
    img1 = img1[:, :, None, ...]

    rep = nfr if neighbor_fr == 'all' else 2*neighbor_fr+1
    img2 = torch.repeat_interleave(volume[:, slc, ...][:, None, ...], rep, 1)
    img2 = img2[:, :, None, ...]

    return img1, img2


def crop_center2d_torch(imgs, crop_size_x_y):
    cropx, cropy = crop_size_x_y[0], crop_size_x_y[1]
    assert imgs.shape[-2] >= cropx and imgs.shape[-1] >= cropy
    shape_x, shape_y = imgs.shape[-2], imgs.shape[-1]
    startx = shape_x // 2 - (cropx // 2)
    starty = shape_y // 2 - (cropy // 2)

    imgs = imgs[..., startx:startx + cropx, starty:starty + cropy]
    return imgs
