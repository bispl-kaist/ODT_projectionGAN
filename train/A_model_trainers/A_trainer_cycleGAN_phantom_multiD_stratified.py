import torch
from torch import nn, optim, multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
from collections import defaultdict

from utils.run_utils import get_logger
from utils.train_utils import CheckpointManager, make_grid_triplet, make_k_grid, make_input_triplet, make_grid_doublet, \
                            make_input_RSS, make_RSS, imsave, make_recons, ImagePool, get_scale_weights, stratify_images
from utils.train_utils_gan import cycleGANCheckpointManager_comparison

from metrics.my_ssim import ssim_loss
from metrics.new_1d_ssim import SSIM
from metrics.custom_losses import psnr, nmse

from data.data_transforms import root_sum_of_squares, fake_input_gen, \
    normalize_im, nchw_to_kspace, fft2


class ModelTrainerIMGgan:
    """
    Model Trainer for k-space learning or complex image learning
    with losses in complex image domains and real valued image domains.
    All learning occurs in k-space or complex image domains
    while all losses are obtained from either complex images or real-valued images.
    """

    def __init__(self, args, modelG_DtoF, modelG_FtoD, modelD_Full, modelD_Down, train_loader,
                 optimizerG_DtoF,  optimizerG_FtoD, optimizerD_Full, optimizerD_Down,
                 input_train_transform, output_transform, losses, schedulerG_DtoF=None, schedulerG_FtoD=None,
                 schedulerD_Full=None, schedulerD_Down=None):

        multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)
        # Checking whether inputs are correct.
        assert isinstance(modelG_DtoF, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(optimizerG_DtoF, optim.Optimizer), '`optimizer` must be a Pytorch Optimizer.'
        assert isinstance(train_loader, DataLoader), \
            '`train_loader` must be Pytorch DataLoader objects.'

        assert callable(input_train_transform), \
            'input_transforms must be callable functions.'
        # I think this would be best practice.
        assert isinstance(output_transform, nn.Module), '`output_transform` must be a Pytorch Module.'

        # 'losses' is expected to be a dictionary.
        losses = nn.ModuleDict(losses)

        if schedulerG_DtoF is not None:
            if isinstance(schedulerG_DtoF, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_scheduler = True
            elif isinstance(schedulerG_DtoF, optim.lr_scheduler._LRScheduler):
                self.metric_scheduler = False
            else:
                raise TypeError('`scheduler` must be a Pytorch Learning Rate Scheduler.')

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.display_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(train_loader.dataset) // (args.display_images * args.batch_size))

        self.checkpointer = cycleGANCheckpointManager_comparison(modelG_DtoF, modelG_FtoD, modelD_Full, modelD_Down,
                                                          optimizerG_DtoF, optimizerG_FtoD, optimizerD_Full,
                                                          optimizerD_Down, mode='min',
                                                          save_best_only=args.save_best_only, ckpt_dir=args.ckpt_path,
                                                          max_to_keep=args.max_to_keep)
        # loading from checkpoint if specified.
        if vars(args).get('load_ckpt'):
            self.checkpointer.load(args.prev_model_ckpt_G_DtoF, args.prev_model_ckpt_G_FtoD, args.prev_model_ckpt_D_Full,
                                   args.prev_model_ckpt_D_Down, load_optimizer=False)

        self.name = args.name
        self.modelG_DtoF = modelG_DtoF
        self.modelG_FtoD = modelG_FtoD
        self.modelD_Full = modelD_Full
        self.modelD_Down = modelD_Down
        self.optimizerG_DtoF = optimizerG_DtoF
        self.optimizerG_FtoD = optimizerG_FtoD
        self.optimizerD_Full = optimizerD_Full
        self.optimizerD_Down = optimizerD_Down

        self.train_loader = train_loader
        self.input_train_transform = input_train_transform
        self.output_transform = output_transform
        self.losses = losses
        self.schedulerG_DtoF = schedulerG_DtoF
        self.schedulerG_FtoD = schedulerG_FtoD
        self.schedulerD_Full = schedulerD_Full
        self.schedulerD_Down = schedulerD_Down

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.use_slice_metrics = args.use_slice_metrics
        self.gan_mode = args.gan_mode
        self.down_scale = args.down_scale
        self.GAN_lambda = torch.tensor(args.GAN_lambda, dtype=torch.float32, device=args.device)
        self.GAN_lambda2 = torch.tensor(args.GAN_lambda2, dtype=torch.float32, device=args.device)
        self.init_GAN_lambda = torch.tensor(args.init_GAN_lambda, dtype=torch.float32, device=args.device)
        self.use_ident = args.use_ident
        self.ident_lambda = torch.tensor(args.ident_lambda, dtype=torch.float32, device=args.device)
        self.stratify_level = args.stratify_level
        self.stratify_lambda = torch.tensor(args.stratify_lambda, dtype=torch.float32, device=args.device)

        self.writer = SummaryWriter(str(args.log_path))
        self.ssim = SSIM(filter_size=7).to(device=args.device)  # Needed to cache the kernel.

        # Axial proj
        self.patch_size = args.patch_size

        self.D_step = args.D_step
        self.use_gp = args.use_gp
        self.clip_limit = args.clip_limit
        self.pool_size = args.pool_size
        self.fake_full_pool = ImagePool(self.pool_size)
        self.fake_down_pool = ImagePool(self.pool_size)

    def train_model(self):
        tic_tic = time()
        self.logger.info(self.name)
        self.logger.info('Beginning Training Loop.')
        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing
            self.epoch = epoch
            # Training
            tic = time()
            train_epoch_G_loss, train_epoch_D_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, train_epoch_G_loss, train_epoch_D_loss,
                                    train_epoch_metrics, elapsed_secs=toc, training=True, verbose=True)

            self.checkpointer.save(metric=train_epoch_G_loss, verbose=True)

            if self.schedulerG_DtoF is not None:
                self.schedulerG_DtoF.step()
                self.schedulerG_FtoD.step()
                self.schedulerD_Full.step()
                self.schedulerD_Down.step()

        # Finishing Training Loop
        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def _train_epoch(self, epoch):
        self.modelG_DtoF.train()
        self.modelG_FtoD.train()
        self.modelD_Full.train()
        self.modelD_Down.train()
        torch.autograd.set_grad_enabled(True)

        epoch_G_loss = list()  # Appending values to list due to numerical underflow.
        epoch_D_loss = list()
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.train_loader, start=1)
        if not self.verbose:  # tqdm has to be on the outermost iterator to function properly.
            data_loader = tqdm(data_loader, total=len(self.train_loader.dataset) // self.batch_size)

        # img_cs -- scaled
        # img_full -- needs to be scaled after masking
        for step, data in data_loader:
            with torch.no_grad():  # Data pre-processing should be done without gradients.
                img_down, img_full, extra_params = self.input_train_transform(*data)

            # 'recons' is a dictionary containing k-space, complex image, and real image reconstructions.
            s_img_full, outputs_full_cfc, s_img_down, outputs_down_fcf, s_fake_img_full, s_fake_img_down, step_G_loss, step_metrics\
                = self._train_step_G(img_down, img_full, extra_params)
            # Update both discriminator times
            step_D_loss = 0
            for i in range(self.D_step):
                step_D_loss += self._train_step_D_double(img_full, outputs_full_cfc, img_down, outputs_down_fcf)
            epoch_G_loss.append(step_G_loss.detach())  # Perhaps not elegant, but underflow makes this necessary.
            epoch_D_loss.append(step_D_loss.detach())

            # Gradients are not calculated so as to boost speed and remove weird errors.
            # Retrospective scheme : full - cs - full
            # Since we only have the ground truth for this scheme, we should use it to calculate slice metrics
            with torch.no_grad():  # Update epoch loss and metrics
                if self.use_slice_metrics:
                    slice_metrics = self._get_slice_metrics(s_fake_img_full, s_img_full, self.batch_size)
                    step_metrics.update(slice_metrics)
                [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

                if self.verbose:
                    self._log_step_outputs(epoch, step, step_G_loss, step_D_loss, step_metrics, training=True)

                if self.display_interval and (step % self.display_interval == 0):

                    # D F D
                    for i in range(self.stratify_level):
                        img_down_grid = normalize_im(s_img_down[0, i, :, :].squeeze())
                        img_recon_grid = normalize_im(outputs_full_cfc.squeeze())
                        img_redown_grid = normalize_im(s_fake_img_down[0, i, :, :].squeeze())
                        irt_grid_dfd = make_grid_triplet(img_down_grid, img_recon_grid, img_redown_grid)
                        self.writer.add_image(f'DtoF{i}/{step}', irt_grid_dfd, epoch, dataformats='HW')

                    # F D F
                    for i in range(self.stratify_level):
                        img_full_grid = normalize_im(s_img_full[0, i, :, :].squeeze())
                        img_fakedown_grid = normalize_im(outputs_down_fcf.squeeze())
                        img_fakefull_grid = normalize_im(s_fake_img_full[0, i, :, :].squeeze())
                        irt_grid = make_grid_triplet(img_full_grid, img_fakedown_grid, img_fakefull_grid)
                        self.writer.add_image(f'FDF{i}/{step}', irt_grid, epoch, dataformats='HW')

        # Converted to scalar and dict with scalar forms.
        return self._get_epoch_outputs(epoch, epoch_G_loss, epoch_D_loss, epoch_metrics, training=True)

    def _train_step_G(self, img_down, img_full, extra_params):
        for param in self.modelD_Full.parameters():
            param.requires_grad = False
        for param in self.modelD_Down.parameters():
            param.requires_grad = False
        self.optimizerG_DtoF.zero_grad()
        self.optimizerG_FtoD.zero_grad()

        GAN_weights = get_scale_weights(self.epoch, self.num_epochs, self.init_GAN_lambda)
        ## Down - Full - Down ##
        outputs_full_cfc = self.modelG_DtoF(img_down)
        fake_img_down = self.modelG_FtoD(outputs_full_cfc)
        # cyclic loss
        s_img_down, s_fake_img_down = stratify_images(img_down, fake_img_down, self.stratify_level)
        cyclic_loss_cfc = self.losses['cyclic_loss'](s_img_down, s_fake_img_down) + self.losses['cyclic_loss'](img_down, fake_img_down) * self.stratify_lambda
        # GAN loss
        pred_cfc = self.modelD_Full(outputs_full_cfc, GAN_weights)
        D_loss_cfc = self.losses['GAN_loss'](pred_cfc, True)

        step_loss_cfc = cyclic_loss_cfc + self.GAN_lambda * D_loss_cfc

        ## Full - Down - Full ##
        outputs_down_fcf = self.modelG_FtoD(img_full)
        fake_img_full = self.modelG_DtoF(outputs_down_fcf)
        # cyclic loss
        s_img_full, s_fake_img_full = stratify_images(img_full, fake_img_full, self.stratify_level)
        cyclic_loss_fcf = self.losses['cyclic_loss'](img_full, fake_img_full) + self.losses['cyclic_loss'](s_img_full, s_fake_img_full) * self.stratify_lambda
        # GAN loss
        pred_fcf = self.modelD_Down(outputs_down_fcf, GAN_weights)
        D_loss_fcf = self.losses['GAN_loss'](pred_fcf, True)

        step_loss_fcf = cyclic_loss_fcf + self.GAN_lambda2 * D_loss_fcf

        ## Identity loss
        ident_img_full = self.modelG_DtoF(img_full)
        identity_loss = self.losses['cyclic_loss'](img_full, ident_img_full)

        step_metrics = {'cyclic_loss_cfc': cyclic_loss_cfc, 'D_loss_cfc': D_loss_cfc,
                        'cyclic_loss_fcf': cyclic_loss_fcf, 'D_loss_fcf': D_loss_fcf, 'identity_loss': identity_loss}

        step_loss = step_loss_cfc + step_loss_fcf + self.ident_lambda * identity_loss
        step_loss.backward()

        self.optimizerG_DtoF.step()
        self.optimizerG_FtoD.step()

        return s_img_full, outputs_full_cfc, s_img_down, outputs_down_fcf, s_fake_img_full, s_fake_img_down, step_loss, step_metrics

    def _train_step_D_double(self, img_full, outputs_full_cfc, img_down, outputs_down_fcf):
        for param in self.modelD_Full.parameters():
            param.requires_grad = True
        for param in self.modelD_Down.parameters():
            param.requires_grad = True

        self.optimizerD_Full.zero_grad()
        self.optimizerD_Down.zero_grad()

        GAN_weights = get_scale_weights(self.epoch, self.num_epochs, self.init_GAN_lambda)

        # D Full
        # Real
        pred_real_full = self.modelD_Full(img_full, GAN_weights)
        loss_D_real_full = self.losses['GAN_loss'](pred_real_full, True)
        # Fake
        outputs_full_cfc = self.fake_full_pool.query(outputs_full_cfc.detach())
        pred_fake_full = self.modelD_Full(outputs_full_cfc, GAN_weights)
        loss_D_fake_full = self.losses['GAN_loss'](pred_fake_full, False)

        loss_D_Full = (loss_D_real_full + loss_D_fake_full)
        loss_D_Full.backward()
        self.optimizerD_Full.step()

        # D Down
        # Real
        pred_real_down = self.modelD_Down(img_down, GAN_weights)
        loss_D_real_down = self.losses['GAN_loss'](pred_real_down, True)
        # Fake
        outputs_down_fcf = self.fake_down_pool.query(outputs_down_fcf.detach())
        pred_fake_down = self.modelD_Down(outputs_down_fcf, GAN_weights)
        loss_D_fake_down = self.losses['GAN_loss'](pred_fake_down, False)

        loss_D_Down = (loss_D_real_down + loss_D_fake_down)
        loss_D_Down.backward()
        self.optimizerD_Down.step()

        loss_D = loss_D_Full + loss_D_Down

        return loss_D

    def _get_slice_metrics(self, recons, targets, batch_size):
        img_recons = recons.squeeze().detach()  # Just in case.
        img_targets = targets.squeeze().detach()

        if batch_size != 1:
            slice_ssim = 0
            slice_psnr = 0
            slice_nmse = 0
            for i in range(batch_size):
                max_range = img_targets.max() - img_targets.min()
                slice_ssim += self.ssim(img_recons, img_targets)
                slice_psnr += psnr(img_recons, img_targets, data_range=max_range)
                slice_nmse += nmse(img_recons, img_targets)
            slice_ssim /= batch_size
            slice_psnr /= batch_size
            slice_nmse /= batch_size
        else:  # When single batch is implemented

            max_range = img_targets.max() - img_targets.min()
            slice_ssim = self.ssim(img_recons, img_targets)
            slice_psnr = psnr(img_recons, img_targets, data_range=max_range)
            slice_nmse = nmse(img_recons, img_targets)

        slice_metrics = {
            'slice/ssim': slice_ssim,
            'slice/nmse': slice_nmse,
            'slice/psnr': slice_psnr
        }

        return slice_metrics

    def _get_epoch_outputs(self, epoch, epoch_G_loss, epoch_D_loss, epoch_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        epoch_G_loss = torch.stack(epoch_G_loss)
        epoch_D_loss = torch.stack(epoch_D_loss)
        is_finite = torch.isfinite(epoch_G_loss)
        num_nans = (is_finite.size(0) - is_finite.sum()).item()

        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')
            epoch_G_loss = torch.mean(epoch_G_loss[is_finite]).item()
        else:
            epoch_G_loss = torch.mean(epoch_G_loss).item()
            epoch_D_loss = torch.mean(epoch_D_loss).item()

        for key, value in epoch_metrics.items():
            epoch_metric = torch.stack(value)
            is_finite = torch.isfinite(epoch_metric)
            num_nans = (is_finite.size(0) - is_finite.sum()).item()

            if num_nans > 0:
                self.logger.warning(f'Epoch {epoch} {mode} {key}: {num_nans} NaN values present in {num_slices} slices')
                epoch_metrics[key] = torch.mean(epoch_metric[is_finite]).item()
            else:
                epoch_metrics[key] = torch.mean(epoch_metric).item()

        return epoch_G_loss, epoch_D_loss, epoch_metrics

    def _log_step_outputs(self, epoch, step, step_G_loss, step_D_loss, step_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_G_loss.item():.4e}')
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_D_loss.item():.4e}')
        for key, value in step_metrics.items():
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d}: {mode} {key}: {value.item():.4e}')

    def _log_epoch_outputs(self, epoch, epoch_G_loss, epoch_D_loss, epoch_metrics,
                           elapsed_secs, training=True, verbose=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} {mode}. G_loss: {epoch_G_loss:.4e}, D_loss: {epoch_D_loss:.4e},'
                         f'Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode}_epoch_G_loss', scalar_value=epoch_G_loss, global_step=epoch)
        self.writer.add_scalar(f'{mode}_epoch_D_loss', scalar_value=epoch_D_loss, global_step=epoch)

        if verbose:
            for key, value in epoch_metrics.items():
                self.logger.info(f'Epoch {epoch:03d} {mode}. {key}: {value:.4e}')
                self.writer.add_scalar(f'{mode}_epoch_{key}', scalar_value=value, global_step=epoch)