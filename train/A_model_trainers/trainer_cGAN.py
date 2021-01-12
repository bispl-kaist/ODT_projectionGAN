import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
from collections import defaultdict

from utils.run_utils import get_logger
from utils.train_utils import CheckpointManager, make_grid_triplet, make_k_grid, make_input_triplet, make_grid_doublet, \
                            make_input_RSS, make_RSS, imsave, make_recons, ImagePool
from utils.train_utils_gan import GANCheckpointManager

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

    def __init__(self, args, modelG, modelD, train_loader, val_loader, optimizerG, optimizerD,
                 input_train_transform, output_transform, losses, schedulerG=None, schedulerD=None):

        multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)
        # Checking whether inputs are correct.
        assert isinstance(modelG, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(optimizerG, optim.Optimizer), '`optimizer` must be a Pytorch Optimizer.'
        assert isinstance(train_loader, DataLoader), \
            '`train_loader` must be Pytorch DataLoader objects.'

        assert callable(input_train_transform), \
            'input_transforms must be callable functions.'
        # I think this would be best practice.
        assert isinstance(output_transform, nn.Module), '`output_transform` must be a Pytorch Module.'

        # 'losses' is expected to be a dictionary.
        losses = nn.ModuleDict(losses)

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.display_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(train_loader.dataset) // (args.display_images * args.batch_size))

        self.checkpointer = GANCheckpointManager(modelG, modelD, optimizerG, optimizerD, mode='min',
                                                 save_best_only=args.save_best_only, ckpt_dir=args.ckpt_path,
                                                 max_to_keep=args.max_to_keep)
        # loading from checkpoint if specified.
        if vars(args).get('load_ckpt'):
            self.checkpointer.load(args.prev_model_ckpt_G, args.prev_model_ckpt_D, load_optimizer=False)

        self.name = args.name
        self.modelG = modelG
        self.modelD = modelD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_train_transform = input_train_transform
        self.output_transform = output_transform
        self.losses = losses
        self.schedulerG = schedulerG
        self.schedulerD = schedulerD

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.use_slice_metrics = args.use_slice_metrics
        self.gan_mode = args.gan_mode
        self.GAN_lambda = torch.tensor(args.GAN_lambda, dtype=torch.float32, device=args.device)

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
            # Training
            tic = time()
            train_epoch_G_loss, train_epoch_D_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, train_epoch_G_loss, train_epoch_D_loss,
                                    train_epoch_metrics, elapsed_secs=toc, training=True, verbose=True)

            # Validation
            tic = time()
            val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs_val(epoch, val_epoch_loss,
                                        val_epoch_metrics, elapsed_secs=toc, training=False, verbose=True)

            self.checkpointer.save(metric=val_epoch_loss, verbose=True)

            if self.schedulerG is not None:
                self.schedulerG.step()
                self.schedulerD.step()

        # Finishing Training Loop
        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def _train_epoch(self, epoch):
        self.modelG.train()
        self.modelD.train()
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

            img_full, outputs_full, img_down,step_G_loss, step_metrics = self._train_step_G(img_down, img_full, extra_params)
            # Update both discriminator times
            step_D_loss = 0
            for i in range(self.D_step):
                step_D_loss += self._train_step_D(img_full, outputs_full)
            epoch_G_loss.append(step_G_loss.detach())  # Perhaps not elegant, but underflow makes this necessary.
            epoch_D_loss.append(step_D_loss.detach())

            # Retrospective scheme : full - cs - full
            # Since we only have the ground truth for this scheme, we should use it to calculate slice metrics
            with torch.no_grad():  # Update epoch loss and metrics
                if self.use_slice_metrics:
                    slice_metrics = self._get_slice_metrics(outputs_full, img_full, self.batch_size)
                    step_metrics.update(slice_metrics)
                [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

                if self.verbose:
                    self._log_step_outputs(epoch, step, step_G_loss, step_D_loss, step_metrics, training=True)

                if self.display_interval and (step % self.display_interval == 0):

                    # D to F
                    img_down_grid = normalize_im(img_down.squeeze())
                    img_recon_grid = normalize_im(outputs_full.squeeze())
                    img_label_grid = normalize_im(img_full.squeeze())
                    irl_grid = make_grid_triplet(img_down_grid, img_recon_grid, img_label_grid)
                    self.writer.add_image(f'Train/{step}', irl_grid, epoch, dataformats='HW')

        # Converted to scalar and dict with scalar forms.
        return self._get_epoch_outputs(epoch, epoch_G_loss, epoch_D_loss, epoch_metrics, training=True)

    def _train_step_G(self, img_down, img_full, extra_params):
        for param in self.modelD.parameters():
            param.requires_grad = False
        self.optimizerG.zero_grad()

        ########################
        outputs_full = self.modelG(img_down)
        img_loss = self.losses['img_loss'](outputs_full, img_full)

        pred = self.modelD(outputs_full)
        D_loss = self.losses['GAN_loss'](pred, True)

        step_loss = img_loss + self.GAN_lambda * D_loss
        step_loss.backward()

        step_metrics = {'img_loss': img_loss, 'D_loss': D_loss}
        self.optimizerG.step()

        return img_full, outputs_full, img_down, step_loss, step_metrics

    def _train_step_D(self, img_full, outputs_full):
        for param in self.modelD.parameters():
            param.requires_grad = True

        self.optimizerD.zero_grad()

        # D
        # Real
        pred_real_full = self.modelD(img_full)
        loss_D_real_full = self.losses['GAN_loss'](pred_real_full, True)
        # Fake
        outputs_full_cfc = self.fake_full_pool.query(outputs_full.detach())
        pred_fake_full = self.modelD(outputs_full_cfc)
        loss_D_fake_full = self.losses['GAN_loss'](pred_fake_full, False)

        loss_D_Full = (loss_D_real_full + loss_D_fake_full) * 0.5
        loss_D_Full.backward()
        self.optimizerD.step()

        loss_D = loss_D_Full

        return loss_D

    def _val_epoch(self, epoch):
        self.modelG.eval()
        self.modelD.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_G_loss = list()  # Appending values to list due to numerical underflow.
        epoch_D_loss = list()
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.val_loader, start=1)
        if not self.verbose:  # tqdm has to be on the outermost iterator to function properly.
            data_loader = tqdm(data_loader, total=len(self.val_loader.dataset) // self.batch_size)

        # img_cs -- scaled
        # img_full -- needs to be scaled after masking
        for step, data in data_loader:
            with torch.no_grad():  # Data pre-processing should be done without gradients.
                img_down, img_full, extra_params = self.input_train_transform(*data)

            img_full, outputs_full, img_down, step_G_loss, step_metrics = self._val_step_G(img_down, img_full, extra_params)
            epoch_G_loss.append(step_G_loss.detach())  # Perhaps not elegant, but underflow makes this necessary.

            # Retrospective scheme : full - cs - full
            # Since we only have the ground truth for this scheme, we should use it to calculate slice metrics
            with torch.no_grad():  # Update epoch loss and metrics
                if self.use_slice_metrics:
                    slice_metrics = self._get_slice_metrics(outputs_full, img_full, self.batch_size)
                    step_metrics.update(slice_metrics)
                [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

                if self.verbose:
                    self._log_step_outputs_val(epoch, step, step_G_loss, step_metrics, training=True)

                if self.display_interval and (step % self.display_interval == 0):

                    # D to F
                    img_down_grid = normalize_im(img_down.squeeze())
                    img_recon_grid = normalize_im(outputs_full.squeeze())
                    img_label_grid = normalize_im(img_full.squeeze())
                    irl_grid = make_grid_triplet(img_down_grid, img_recon_grid, img_label_grid)
                    self.writer.add_image(f'Val/{step}', irl_grid, epoch, dataformats='HW')

        # Converted to scalar and dict with scalar forms.
        return self._get_epoch_outputs_val(epoch, epoch_G_loss, epoch_metrics, training=True)

    def _val_step_G(self, img_down, img_full, extra_params):
        for param in self.modelD.parameters():
            param.requires_grad = False

        ########################
        outputs_full = self.modelG(img_down)
        img_loss = self.losses['img_loss'](outputs_full, img_full)

        pred = self.modelD(outputs_full)
        D_loss = self.losses['GAN_loss'](pred, True)

        step_loss = img_loss + self.GAN_lambda * D_loss
        step_metrics = {'img_loss': img_loss, 'D_loss': D_loss}

        return img_full, outputs_full, img_down, step_loss, step_metrics

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


    def _get_epoch_outputs_val(self, epoch, epoch_G_loss, epoch_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        epoch_G_loss = torch.stack(epoch_G_loss)
        is_finite = torch.isfinite(epoch_G_loss)
        num_nans = (is_finite.size(0) - is_finite.sum()).item()

        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')
            epoch_G_loss = torch.mean(epoch_G_loss[is_finite]).item()
        else:
            epoch_G_loss = torch.mean(epoch_G_loss).item()

        for key, value in epoch_metrics.items():
            epoch_metric = torch.stack(value)
            is_finite = torch.isfinite(epoch_metric)
            num_nans = (is_finite.size(0) - is_finite.sum()).item()

            if num_nans > 0:
                self.logger.warning(f'Epoch {epoch} {mode} {key}: {num_nans} NaN values present in {num_slices} slices')
                epoch_metrics[key] = torch.mean(epoch_metric[is_finite]).item()
            else:
                epoch_metrics[key] = torch.mean(epoch_metric).item()

        return epoch_G_loss, epoch_metrics


    def _log_step_outputs(self, epoch, step, step_G_loss, step_D_loss, step_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_G_loss.item():.4e}')
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_D_loss.item():.4e}')
        for key, value in step_metrics.items():
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d}: {mode} {key}: {value.item():.4e}')

    def _log_step_outputs_val(self, epoch, step, step_G_loss, step_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_G_loss.item():.4e}')
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

    def _log_epoch_outputs_val(self, epoch, epoch_loss, epoch_metrics,
                              elapsed_secs, training=True, verbose=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} {mode}. G_loss: {epoch_loss:.4e},'
                         f'Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode}_epoch_G_loss', scalar_value=epoch_loss, global_step=epoch)

        if verbose:
            for key, value in epoch_metrics.items():
                self.logger.info(f'Epoch {epoch:03d} {mode}. {key}: {value:.4e}')
                self.writer.add_scalar(f'{mode}_epoch_{key}', scalar_value=value, global_step=epoch)