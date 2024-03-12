"""
This script is modified from RAFT: https://github.com/princeton-vl/RAFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graft.update import BasicUpdateBlock, SmallUpdateBlock
from models.graft.extractor import BasicEncoder, SmallEncoder
from models.graft.corr import CorrBlock, AlternateCorrBlock
from utils import coords_grid, upflow, switch2frameblock, pad, unpad
import torch.utils.checkpoint as checkpoint

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class TemplateFormer(nn.Module):
    def __init__(self, ch_num=[64, 32, 1], circular=3, average_init=True):
        super(TemplateFormer, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, (3, 1, 1), 1, 1)
        layers = []
        ch_num = [1] + ch_num
        for i, ch in enumerate(ch_num):
            conv = nn.Conv1d(ch, ch_num[i+1], (circular*2+1, 1, 1), 1, (circular, 0, 0))
            if average_init:
                torch.nn.init.constant_(conv.weight, 1/(circular*2+1))
            layers.append(conv)
            layers.append(nn.ReLU())
            if i == len(ch_num) - 2:
                layers = layers[:-1]
                break
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        x = self.seq(x[:, None, ...]).mean(dim=2, keepdim=True)
        return x[:, 0, ...]


class GRAFT(nn.Module):
    def __init__(self, args):
        super(GRAFT, self).__init__()
        self.args = args
        self.group_frameblock_mode = args.frameblock
        self.feature_dim_replicate = args.feature_dim_replicate
        self.iters = args.iters
        self.cnet_channel_num = 1 if not self.group_frameblock_mode else 3
        if self.group_frameblock_mode:
            self.fnet_channel_num = 3 if self.feature_dim_replicate else 1
        else:
            self.fnet_channel_num = 1

        if args.graft_small:
            self.hidden_dim = hdim = 84
            self.context_dim = cdim = 84
            args.corr_levels = 4
            args.corr_radius = 4

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        # if 'dropout' not in self.args:
        self.args.dropout = 0

        # if 'alternate_corr' not in self.args:
        self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.graft_small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, ch_num=self.fnet_channel_num)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout,
                                     ch_num=self.cnet_channel_num)
            input_dim = 84 if args.graft_small else 128
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=input_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 4, W // 4).to(img.device)
        coords1 = coords_grid(N, H // 4, W // 4).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, factor=8):
        """ Upsample flow field [H/factor, W/factor, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, factor * H, factor * W)

    def forward(self, image1, image2, flow_init=None, iters=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        if not iters: iters = self.iters

        if self.group_frameblock_mode: image1 = switch2frameblock(image1)
        [b, f, c1, h, w] = image1.shape
        assert b == 1, 'currently group_mode supports only in condition of batch_size=1'
        # todo: how to avoid reshape and batch must = 1? maybe a new network?
        c2 = image2.shape[2]
        image1 = torch.reshape(image1, (b * f, c1, h, w))
        image2 = torch.reshape(image2, (b * f, c2, h, w))

        image1, unpad_info = pad(image1, divisor=4)
        image2, _ = pad(image2, divisor=4)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1_fnet = image1.clone()
        if self.group_frameblock_mode:
            if self.feature_dim_replicate:
                image2 = torch.repeat_interleave(image2, 3, 1)
            else:
                image1_fnet = image1[:, :1, ...]

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        image1_fnet = image1_fnet.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast():
            fmap1, fmap2 = self.fnet([image1_fnet, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast():
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:

            flow_pad, _ = pad(flow_init, divisor=4)
            flow_pad = upflow(flow_pad, factor=1/4)
            coords1 = coords1 + flow_pad

        flow_predictions = []
        # flow_predictions = torch.zeros((iters, image2.shape[0], 2, image2.shape[-2], image2.shape[-1])).cuda()
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast():
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            # up_mask = None
            if up_mask is None:
                flow_up = upflow(coords1 - coords0, factor=4)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask, factor=4)

            flow_up = unpad(flow_up, **unpad_info)
            # flow_predictions[itr] = flow_up
            # self.flow_prediction[itr] = flow_up
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
