import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.run_utils import get_logger
from metrics.new_1d_ssim import SSIM

from data.data_transforms import fake_input_gen, nchw_to_kspace


from scipy.io import loadmat, savemat


class Infer_CU:

    def __init__(self, args, modelG, eval_loader, eval_input_transform, eval_output_transform):

        multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(modelG, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(eval_loader, DataLoader),'`train_loader` and `val_loader` must be Pytorch DataLoader objects.'

        assert callable(eval_input_transform) and callable(eval_output_transform), \
            'input/output_transforms must be callable functions.'

        self.name = args.name
        self.modelG = modelG
        self.eval_loader = eval_loader
        self.eval_input_transform = eval_input_transform
        self.eval_output_transform = eval_output_transform

        self.batch_size = args.batch_size
        self.writer = SummaryWriter(str(args.log_path))
        self.ssim = SSIM(filter_size=7).to(device=args.device)  # Needed to cache the kernel.
        self.verbose = args.verbose

        self.depth = args.depth
        self.center_slice = self.depth // 2

    def inference_axial_from_coronal(self, args):
        self.logger.info('Starting inference')
        self.modelG.eval()
        torch.autograd.set_grad_enabled(False)

        data_loader = enumerate(self.eval_loader, start=1)
        if not self.verbose:
            data_loader = tqdm(data_loader, total=len(self.eval_loader.dataset))

        save_fdir_upper = Path(args.save_fdir)
        save_fdir_upper.mkdir(exist_ok=True)

        for step, data in data_loader:
            data_0 = data[0].to(args.device).float()
            data_1 = data[1]
            # mkdir save_dir
            full_fname = data_1[0]
            full_fname_list = full_fname.split('/')

            specimen_type = full_fname_list[-3]
            specimen_fname = full_fname_list[-2]
            proj_fname = full_fname_list[-1]

            save_fdir_axi = save_fdir_upper / 'recon_axi'
            save_fdir_axi.mkdir(exist_ok=True)

            save_fdir_axi_type = save_fdir_axi / specimen_type
            save_fdir_axi_type.mkdir(exist_ok=True)

            save_fdir_axi_specimen = save_fdir_axi_type / specimen_fname
            save_fdir_axi_specimen.mkdir(exist_ok=True)

            # Inferecne stage
            input_axi_slice, extra_params = self.eval_input_transform(data_0, data_1)

            axi_outputs = self.modelG(input_axi_slice)
            axi_recons = self.eval_output_transform(axi_outputs, extra_params)

            fname = proj_fname[:-4]  # 'img_seg1_x136'

            save_axi_recons = axi_recons.detach().cpu().numpy()
            save_axi_recons_dict = {'recons': save_axi_recons}
            save_axi_fname = str(save_fdir_axi_specimen) + '/' + fname + '.mat'
            savemat(save_axi_fname, save_axi_recons_dict)