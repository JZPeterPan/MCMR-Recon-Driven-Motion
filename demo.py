import numpy as np, yaml, torch, os, sys, pathlib, glob
from tqdm import tqdm
from utils import dict2obj, brightness_augmentation_torch, normalize_3d_torch, count_parameters
from models import MocoRecon, ReconDCPM
from losses import CriterionMotion, CriterionRecon


def create_dummy_data(nCoil, nFrame, nX, nY):
    kspace = torch.randn(1, nCoil, nFrame, nX, nY).cuda() + 1j * torch.randn(1, nCoil, nFrame, nX, nY).cuda()
    csm = torch.randn(1, nCoil, 1, nX, nY).cuda() + 1j * torch.randn(1, nCoil, 1, nX, nY).cuda()
    mask = torch.ones(1, 1, nFrame, 1, nY).cuda()
    ref = torch.randn(1, nFrame, nX, nY).cuda()
    return kspace, csm, mask, ref


class TrainerMocoRecon:
    def __init__(self, config):
        super().__init__()
        self.config = config.general
        self.experiment_dir = os.path.join(config.general.exp_save_root, config.general.exp_name)
        # if not config.general.debug:
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.num_epochs = config.training.num_epochs

        # define data loader
        self.dummy_train_loader = [1,2,3]
        self.dummy_test_loader = [1,2,3]

        # network
        self.network = MocoRecon(config.network)
        self.network.cuda()
        self.pre_denoising = ReconDCPM(**config.network.DCPM.__dict__).cuda()

        print("Parameter Count: %d" % count_parameters(self.network))
        if config.training.restore_ckpt: self.restore_weights(config.training)

        # optimizer and scheduler
        self.optimizer = eval(f'torch.optim.{config.optimizer.which}')(self.network.parameters(), **eval(f'config.optimizer.{config.optimizer.which}').__dict__)
        self.scheduler = eval(f'torch.optim.lr_scheduler.{config.scheduler.which}')(self.optimizer, steps_per_epoch=len(self.dummy_train_loader)+1, **eval(f'config.scheduler.{config.scheduler.which}').__dict__)

        self.loss_scaler = torch.cuda.amp.GradScaler(enabled=config.training.use_mixed_precision)

        self.train_criterion_flow = CriterionMotion(config.train_loss_motion)
        self.train_criterion_recon = CriterionRecon(config.train_loss_recon)

        self.eval_criterion_flow = CriterionMotion(config.eval_loss_motion)
        self.eval_criterion_recon = CriterionRecon(config.eval_loss_recon)

        self.recon_frame_amount_train = self.config.recon_frame_amount_train
        self.recon_frame_amount_val = self.config.recon_frame_amount_val
        self.BP_recon_w = config.general.weighting_recon
        self.BP_motion_w = config.general.weighting_motion

    def backwards(self, loss):
        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update()

    def save_model(self, epoch):
        save_dict = {'epoch': epoch,
                     'model_state_dict': self.network.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'scheduler_state_dict': self.scheduler.scheduler.state_dict(),
                     }
        torch.save(save_dict, f'{self.experiment_dir}/model_{epoch+1:03d}.pth')

    def restore_weights(self, args):
        if os.path.isdir(args.restore_ckpt):
            args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
        ckpt = torch.load(args.restore_ckpt)
        self.network.load_state_dict(ckpt['model_state_dict'], strict=True)

    def run(self):
        pbar = tqdm(range(self.start_epoch, self.num_epochs))
        for epoch in pbar:
            self.network.train()
            for sidx, batch in enumerate(self.dummy_train_loader):
                kspace, csm, mask, ref = create_dummy_data(30, self.config.recon_frame_amount_train, 192, 156)
                init = self.pre_denoising(img=None, kspace=kspace, mask=mask, smaps=csm)

                volume = brightness_augmentation_torch(normalize_3d_torch(init.abs().type(torch.float32)))

                ref_normal = normalize_3d_torch(ref.abs(), scale=255).type(torch.uint8).type(torch.float32)

                estimate_frames = np.sort(np.random.choice(volume.shape[1], self.recon_frame_amount_train, replace=False))
                self.optimizer.zero_grad()
                recon_im, motion_n2m_list, _ = self.network(init.detach(), volume, kspace, mask, csm, estimate_frames, self.config.recon_neighbor_frames_train)
                ref_n2m_list = self.network.get_groupwise_imgs(ref_normal, estimate_frames, recon_neighbor_frames=self.config.recon_neighbor_frames_train)

                recon_loss = self.train_criterion_recon(recon_im, ref[:, estimate_frames, ...])['total_loss']

                #  optional conventional motion warping loss
                motion_loss = [self.train_criterion_flow(flow, *ims)['total_loss'] for flow, ims in zip(motion_n2m_list, ref_n2m_list)]
                motion_loss = [m for m in motion_loss if m.requires_grad]
                motion_loss = 1/len(motion_loss)*sum(motion_loss)

                self.backwards(self.BP_motion_w * motion_loss + self.BP_recon_w * recon_loss)
                self.scheduler.step()

                # log ...

            self.network.eval()
            for sidx, batch in enumerate(self.dummy_test_loader):
                kspace, csm, mask, ref = create_dummy_data(30, self.config.recon_frame_amount_val, 192, 156)
                ref_normal = normalize_3d_torch(ref.abs(), scale=255).type(torch.uint8).type(torch.float32)
                init = self.pre_denoising(img=None, kspace=kspace, mask=mask, smaps=csm)

                volume = normalize_3d_torch(init.abs().type(torch.float32))
                estimate_frames = np.arange(self.recon_frame_amount_val)
                with torch.no_grad():
                    recon_im, motion_n2m_list, _ = self.network(init, volume, kspace, mask, csm, estimate_frames, self.config.recon_neighbor_frames_val)
                    ref_n2m_list = self.network.get_groupwise_imgs(ref_normal, estimate_frames, recon_neighbor_frames=self.config.recon_neighbor_frames_val)

                    recon_loss = self.eval_criterion_recon(recon_im, ref[:, estimate_frames])

                    #  optional conventional motion warping loss
                    motion_loss = [self.eval_criterion_flow(flow, *ims)['total_loss'] for flow, ims in zip(motion_n2m_list, ref_n2m_list)]
                    motion_loss = 1 / len(motion_loss) * sum(motion_loss)

                # motion_sample = torch.cat([motion[-1][1][None] for motion in motion_n2m_list])

                # log ...

            if epoch % self.config.weights_save_frequency == 0:
                self.save_model(epoch)


if __name__ == '__main__':
    # load config file
    config = 'MocoRecon.yaml'
    with open(config) as f:
        print(f'Using {config} as config file')
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    # set random seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(config.general.seed)
    np.random.seed(config.general.seed)
    torch.cuda.manual_seed_all(config.general.seed)

    # define network
    trainer = TrainerMocoRecon(config)
    trainer.run()