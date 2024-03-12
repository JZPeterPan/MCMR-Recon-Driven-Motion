import torch
from utils.loss_related import gradient, gradient_central, temporal_grad_central, jacobian_determinant
import optoth.warp
import lpips


class CriterionBase(torch.nn.Module):
    def __init__(self, config, mask=None):
        super(CriterionBase, self).__init__()
        self.loss_names = config.which
        self.loss_weights = config.loss_weights
        self.mask = mask
        self.loss_list = []

        for loss_name in config.which:
            loss_args = eval(f'config.{loss_name}').__dict__
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def get_loss(self, loss_name, args_dict):
        if loss_name == 'photometric':
            return PhotometricLoss(**args_dict)
        elif loss_name == 'sp_smooth':
            return SpatialSmooth(**args_dict)
        elif loss_name == 'temp_smooth':
            assert self.group_mode, 'temp_smooth can only be used in group_mode'
            return TemporalSmooth(**args_dict)
        elif loss_name == 'psnr':
            return PSNR(**args_dict)
        elif loss_name == 'LPIPS':
            return LPIPSLoss(**args_dict)
        else:
            raise NotImplementedError


class JacobianLoss(torch.nn.Module):
    def __init__(self):
        super(JacobianLoss, self).__init__()

    def forward(self, flow_preds):
        flow = flow_preds[-1]
        flow = flow.permute(0, 2, 3, 1)
        loss_jac_pos_percentage = jacobian_determinant(flow)
        return loss_jac_pos_percentage


class CriterionMotion(CriterionBase, torch.nn.Module):
    def __init__(self, config, mask=None):
        # self.config = config
        self.group_mode = config.group_mode
        self.iterative_losses = config.iterative_losses
        self.iteration_gamma = config.iteration_gamma
        super().__init__(config=config, mask=mask)

    def forward(self, flow_preds, image1, image2, valid_mask=None):
        total_iters = len(flow_preds)

        if self.group_mode:
            assert image1.shape[0] == image2.shape[0] == 1, 'currently group_mode supports only in condition of batch_size=1'
            image1 = torch.reshape(image1, (image1.shape[0] * image1.shape[1], *image1.shape[2:]))
            image2 = torch.reshape(image2, (image2.shape[0] * image2.shape[1], *image2.shape[2:]))
        loss_dict = {}
        total_loss = 0
        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            partial_loss = 0
            if loss_name in self.iterative_losses:
                i_weights = [self.iteration_gamma ** (total_iters - iteration - 1) for iteration in range(total_iters)]
                flow_in_loop = flow_preds.copy()
            else:
                i_weights = [1]
                flow_in_loop = [flow_preds[-1]]
            for i, (i_weight, i_flow) in enumerate(zip(i_weights, flow_in_loop)):
                if loss_name == 'photometric':
                    warped = optoth.warp.WarpFunction.apply(image1, i_flow.permute(0, 2, 3, 1).flip(-1))
                    i_loss = loss_term(image2, warped)
                elif loss_name == 'sp_smooth':
                    i_loss = loss_term(i_flow, image1)
                elif loss_name == 'temp_smooth':
                    i_loss = loss_term(i_flow)
                else:
                    raise KeyError('loss_name not registered')
                partial_loss += i_weight * loss_weight * i_loss

            loss_dict[loss_name] = partial_loss
            total_loss += partial_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict


class CriterionRecon(CriterionBase, torch.nn.Module):
    def __init__(self, config, mask=None):
        super().__init__(config=config, mask=mask)
        self.cardiac_crop_quantitative_metric = config.cardiac_crop_quantitative_metric

    def forward(self, recon, ref, valid_mask=None, mask_boundary=None):
        loss_dict = {}
        total_loss = 0

        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            if loss_name == 'photometric' or loss_name == 'psnr' or loss_name == 'LPIPS':
                partial_loss = loss_weight * loss_term(recon, ref)
            else:
                raise KeyError('loss_name not registered')
            loss_dict[loss_name] = partial_loss
            total_loss += partial_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict


class PhotometricLoss(torch.nn.Module):
    def __init__(self, mode):
        super(PhotometricLoss, self).__init__()
        assert mode in ('charbonnier', 'L1', 'L2')
        if mode == 'charbonnier':
            self.loss = CharbonnierLoss()
        elif mode == 'L1':
            self.loss = torch.nn.L1Loss(reduction='mean')
        elif mode == 'L2':
            self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, outputs):
        return self.loss(inputs, outputs)


class LPIPSLoss(torch.nn.Module):
    def __init__(self, net_type='alex', data_convert=True, detach=True):
        super(LPIPSLoss, self).__init__()
        self.loss = lpips.LPIPS(net=net_type).cuda()
        self.data_convert = data_convert
        self.detach = detach

    def forward(self, inputs, outputs):
        if self.data_convert:
            inputs = self.data_preprocess(inputs)
            outputs = self.data_preprocess(outputs)
        loss = self.loss(inputs, outputs)
        if self.detach:
            loss = loss.detach()
        return loss.sum()

    def data_preprocess(self, image_ch1):
        image_ch1 = image_ch1.abs()
        image_ch1 = image_ch1/torch.max(image_ch1)*2-1
        image_ch3 = torch.cat([image_ch1, image_ch1, image_ch1], dim=0)
        image_ch3 = image_ch3.permute(1, 0, 2, 3)
        return image_ch3


class PSNR(torch.nn.Module):
    def __init__(self, magnitude_psnr=True):
        super(PSNR, self).__init__()
        self.magnitude_psnr = magnitude_psnr

    def forward(self, u, g):
        """

        :param u: noised image
        :param g: ground-truth image
        :param max_value:
        :return:
        """
        if self.magnitude_psnr:
            u, g = torch.abs(u), torch.abs(g)
        batch_size = u.shape[0]
        diff = (u.reshape(batch_size, -1) - g.reshape(batch_size, -1))
        square = torch.conj(diff) * diff
        max_value = g.abs().max()
        if square.is_complex():
            square = square.real
        v = torch.mean(20 * torch.log10(max_value / torch.sqrt(torch.mean(square, -1))))
        return v


class TemporalSmooth(torch.nn.Module):
    def __init__(self, mode, grad):
        super(TemporalSmooth, self).__init__()
        assert mode in ('forward', 'central')
        assert grad in (1, 2)
        self.mode = mode
        self.grad = grad

    def forward(self, flow):
        dt = flow[1:, ...] - flow[:-1, ...] if self.mode == 'forward' else temporal_grad_central(flow)
        if self.grad == 2:
            dt = dt[1:, ...] - dt[:-1, ...] if self.mode == 'forward' else temporal_grad_central(dt)

        eps = 1e-6
        dt = torch.sqrt(dt**2 + eps)
        return dt.mean()/2


class SpatialSmooth(torch.nn.Module):
    def __init__(self, mode, grad, boundary_awareness):
        super(SpatialSmooth, self).__init__()
        assert mode in ('forward', 'central')
        assert grad in (1, 2)
        if mode == 'forward':
            self.loss = SpatialSmoothForward(grad=grad, boundary_awareness=boundary_awareness)
        elif mode == 'central':
            self.loss = SpatialSmoothCentral(grad=grad, boundary_awareness=boundary_awareness)

    def forward(self, flow, image=None):
        return self.loss(flow, image)


class SpatialSmoothForward(torch.nn.Module):
    def __init__(self, grad, boundary_awareness, boundary_alpha=10):
        super(SpatialSmoothForward, self).__init__()
        self.grad = grad
        self.boundary_awareness = boundary_awareness
        self.boundary_alpha = boundary_alpha

    def forward(self, flow, image=None):
        dx, dy = gradient(flow)
        if self.grad == 1:
            final_x, final_y = dx.abs(), dy.abs()
            # eps = 1e-6
            # final_x, final_y = torch.sqrt(dx**2 + eps), torch.sqrt(dy**2 + eps)
        elif self.grad == 2:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            final_x, final_y = dx2.abs(), dy2.abs()
            # eps = 1e-6
            # final_x, final_y = torch.sqrt(dx2**2 + eps), torch.sqrt(dy2**2 + eps)

        if self.boundary_awareness:
            img_dx, img_dy = gradient(image)
            weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * self.boundary_alpha)
            weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * self.boundary_alpha)
            loss_x = weights_x * final_x / 2.
            loss_y = weights_y * final_y / 2.
        else:
            loss_x = final_x / 2.
            loss_y = final_y / 2.

        return loss_x.mean() / 2. + loss_y.mean() / 2.


class SpatialSmoothCentral(torch.nn.Module):
    def __init__(self, grad, boundary_awareness, boundary_alpha=10):
        super(SpatialSmoothCentral, self).__init__()
        self.grad = grad
        assert not boundary_awareness
        self.boundary_awareness = boundary_awareness
        self.boundary_alpha = boundary_alpha

    def forward(self, flow, image=None):
        # todo: forward need to be verify!
        grad_x, grad_y = gradient_central(flow)
        if self.grad == 1:
            return (grad_x.square() + grad_y.square()).mean()
        elif self.grad == 2:
            grad_xx, grad_xy = gradient_central(grad_x)
            grad_yx, grad_yy = gradient_central(grad_y)
            return (grad_xx.square() + grad_xy.square() + grad_yx.square() + grad_yy.square()).mean()


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, alpha=0.45):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, y):
        diff = x - y
        square = torch.conj(diff) * diff
        if square.is_complex():
            square = square.real
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.pow(square + self.eps, exponent=self.alpha))
        return loss