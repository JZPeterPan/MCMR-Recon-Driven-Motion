import torch.nn.functional as F
import numpy as np
import pystrum.pynd.ndutils as nd

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def gradient_central(data):
    paddings_x = (1, 1, 0, 0)
    paddings_y = (0, 0, 1, 1)

    pad_x = F.pad(data, paddings_x, mode='replicate')
    pad_y = F.pad(data, paddings_y, mode='replicate')

    grad_x = pad_x[:, :, :, 2:] - 2 * data + pad_x[:, :, :, :-2]
    grad_y = pad_y[:, :, 2:] - 2 * data + pad_y[:, :, :-2]

    return grad_x, grad_y


def temporal_grad_central(data):
    data = data.permute(3, 1, 2, 0)
    padding_t = (1, 1, 0, 0)
    pad_t = F.pad(data, padding_t, mode='replicate')
    grad_t = pad_t[..., 2:] - 2 * data + pad_t[..., :-2]
    grad_t = grad_t.permute(3, 1, 2, 0)
    return grad_t


def jacobian_determinant(flow):
    # check inputs
    J_list = []
    flow = flow.detach().cpu().numpy()
    for i in range(flow.shape[0]):
        flow_i = flow[i]  # .transpose(1, 2, 3, 0)
        volshape = flow_i.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
        # compute grid
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))[..., :2]
        # compute gradients
        J = np.gradient(flow_i + grid)
        # 3D flow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]
            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
            J_det = Jdet0 - Jdet1 + Jdet2
        else:  # must be 2
            dfdx = J[0]
            dfdy = J[1]
            J_det = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
        jac_det_zero = 100 * np.sum(J_det <= 0) / np.prod(volshape)
        J_list.append(jac_det_zero)
    return J_list


def jacobian_determinant_original_version(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]