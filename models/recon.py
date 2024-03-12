import torch
from utils import MulticoilForwardOp, MulticoilAdjointOp, MulticoilMotionForwardOp, MulticoilMotionAdjointOp
from models.cg import DCPM
from utils import neighboring_frame_select, group_mode_img_select_torch
from models.graft.graft import GRAFT


class MocoRecon(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.motion_train_dropout = config.motion_train_dropout if hasattr(config, 'motion_train_dropout') else 0
        assert config.motion_estimator == 'GRAFT', 'Only GRAFT is supported in this repo. You can create your own one.'
        self.motion_estimator = GRAFT(config.GRAFT)

        assert config.dataConsistency == 'DCPMMotion', 'Only DCPMMotion is supported in this repo. You can create your own one.'
        self.recon_block = ReconDCPMMotion(**config.DCPMMotion.__dict__)
        self.recon_motion_detach = False if config.BP_recon else True

    def forward(self, init, volume, kspace, masks, smaps, recon_frames, recon_neighbor_frames='all'):
        flow_n2m_list = []
        init = init[:, recon_frames, ...]

        # dropout_f = np.random.choice(recon_frames, int(self.motion_train_dropout*len(recon_frames)), replace=False)
        flow_recon = torch.zeros([len(recon_frames), volume.shape[1] if recon_neighbor_frames == 'all' else 2*recon_neighbor_frames+1, 2, init.shape[2], init.shape[3]]).cuda()
        for idx, f in enumerate(recon_frames):
            im2, im1 = group_mode_img_select_torch(volume, f, neighbor_fr=recon_neighbor_frames)  # => im1 all the identical img, im2 the neighboring imgs
            flow = self.motion_estimator(im1, im2)  # flow of im1 -> im2
            # if (f in dropout_f) and self.training:  # dropout some frames estimation to save memory in training
            #     flow = [flo.detach() for flo in flow]
            flow_recon[idx, ...] = flow[-1]
            flow_n2m_list.append(flow)

        flow_recon = flow_recon.permute(0, 1, 3, 4, 2).contiguous()

        if self.recon_motion_detach: flow_recon = flow_recon.detach()
        # self2self has to be 0
        flow_recon[:, int(volume.shape[1]/2) if recon_neighbor_frames == 'all' else recon_neighbor_frames, ...] = 0

        recon_im = self.recon_block(init, kspace, masks, smaps, flow_recon.flip(-1), recon_frames, recon_neighbor_frames)

        return recon_im, flow_n2m_list, flow_recon

    def get_groupwise_imgs(self, volume, recon_frames, recon_neighbor_frames='all', abs=True):
        image_n2m_list = []
        for f in recon_frames:
            im2, im1 = group_mode_img_select_torch(volume, f, neighbor_fr=recon_neighbor_frames, abs=abs)  # im1 all the same, im2 different
            image_n2m_list.append([im1, im2])
        return image_n2m_list


class ReconDCPM(torch.nn.Module):
    def __init__(self, max_iter, weight_init=1e12, tol=1e-12):

        super().__init__()
        self.A = MulticoilForwardOp(center=True, coil_axis=-4, channel_dim_defined=False)
        self.AH = MulticoilAdjointOp(center=True, coil_axis=-4, channel_dim_defined=False)

        self.DC = DCPM(self.A, self.AH, weight_init=weight_init, max_iter=max_iter, tol=tol, weight_scale=1.0, requires_grad=False)

    def forward(self, img, kspace, mask, smaps):
        if img is None:
            img = self.AH(kspace, mask, smaps)
        recon_im = self.DC([img, kspace, mask, smaps])
        return recon_im


class ReconDCPMMotion(torch.nn.Module):
    def __init__(self, max_iter, weight_init=1e-12, tol=1e-12):
        super().__init__()
        self.A_motion = MulticoilMotionForwardOp(center=True, coil_axis=-5, channel_dim_defined=False)
        self.AH_motion = MulticoilMotionAdjointOp(center=True, coil_axis=-5, channel_dim_defined=False)

        self.DC = DCPM(self.A_motion, self.AH_motion, weight_init=weight_init, max_iter=max_iter, tol=tol, weight_scale=1.0)

    def forward(self, recon_image, kspace, mask, smaps, flow, recon_frames=None, recon_neighbor_frames='all'):

        # modify the input size to fit the DCPM
        if flow.dim() != 6:
            flow = flow[None, ...]
            assert flow.dim() == 6
        if recon_frames is None:
            recon_frames = range(flow.shape[1])
        if recon_image.dim() != 4:
            recon_image = recon_image[None, ...]
        if kspace.dim() != 6:
            kspace_expand = torch.zeros((1, kspace.shape[1], *flow.shape[1:-1])).type(torch.complex64).cuda()
            for idx, slc in enumerate(recon_frames):
                neighbor_kspace = neighboring_frame_select(kspace, slc, recon_neighbor_frames, frame_dim=2)
                neighbor_kspace = neighbor_kspace[:, :, None, ...]
                kspace_expand[:, :, idx:idx+1] = neighbor_kspace
            kspace = kspace_expand
        if mask.dim() != 6:
            mask_expand = torch.zeros((*mask.shape[:2], *flow.shape[1:-1])).cuda()
            for idx, slc in enumerate(recon_frames):
                neighbor_mask = neighboring_frame_select(mask, slc, recon_neighbor_frames, frame_dim=2)
                neighbor_mask = neighbor_mask[:, :, None, ...]
                mask_expand[:, :, idx:idx+1] = neighbor_mask
            mask = mask_expand
        if smaps.dim() != 6:
            smaps = smaps[:, :, None, ...]

        recon_im = self.DC([recon_image, kspace, mask, smaps, flow])
        return recon_im
