import torch
from torch import nn, optim
from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_cycle_data_loaders_axial_single, create_AAPM_cGAN_data_loaders

from data.input_transforms import Prefetch2Device, PreProcessScale_Axial, PreProcessScale_Rot, PreProcessScale_affine
from data.output_transforms import SingleOutputTransform

from models.fc_unet import Unet, UnetSA_v2, UnetSA_bypass, UnetSA_light
from models.Discriminator import GANLoss, MultiScaleDiscriminator
from train.A_model_trainers.A_trainer_cycleGAN_multiD import ModelTrainerIMGgan


def train_img(args):

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_root)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_root)
    log_path.mkdir(exist_ok=True)

    log_path = log_path
    log_path.mkdir(exist_ok=True)

    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Input transforms. These are on a per-slice basis.
    # UNET architecture requires that all inputs be dividable by some power of 2.
    divisor = 2 ** args.num_pool_layers

    data_prefetch = Prefetch2Device(device)

    # std preprocessing
    input_train_transform = PreProcessScale_Axial(args.device, use_seed=True, divisor=divisor,
                                                  use_patch=args.use_patch, patch_size=args.patch_size)

    # Rotation augmentation
    # input_train_transform = PreProcessScale_Rot(args.device, use_seed=True, divisor=divisor,
    #                                             use_patch=args.use_patch, patch_size=args.patch_size)

    # General Affine augmentation
    # input_train_transform = PreProcessScale_affine(args.device, use_seed=True, divisor=divisor,
    #                                                use_patch=args.use_patch, patch_size=args.patch_size)

    # DataLoaders
    # train_loader = create_cycle_data_loaders_axial_single(args, transform=data_prefetch)
    train_loader = create_AAPM_cGAN_data_loaders(args, transform=data_prefetch)

    losses = dict(
        cyclic_loss=nn.MSELoss(reduction='mean'),
        GAN_loss=GANLoss(args.gan_mode).to(device),
    )

    output_transform = SingleOutputTransform()

    data_chans = args.depth

    modelG_DtoF = Unet(in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
                       use_residual=args.use_residual).to(device)
    modelG_FtoD = Unet(in_chans=data_chans, out_chans=data_chans, chans=16, num_pool_layers=2,
                       use_residual=args.use_residual).to(device)
    # modelG_DtoF = UnetSA_light(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
    #                            num_pool_layers=args.num_pool_layers,
    #                            use_residual=args.use_residual).to(device)
    # modelG_FtoD = UnetSA_light(in_chans=data_chans, out_chans=data_chans, chans=16, num_pool_layers=2,
    #                            use_residual=args.use_residual).to(device)

    modelD_Full = MultiScaleDiscriminator(args.patch_size, max_n_scales=args.down_scale, base_channels=args.chans).to(device)
    modelD_Down = MultiScaleDiscriminator(args.patch_size, max_n_scales=args.down_scale, base_channels=args.chans).to(device)

    optimizerG_DtoF = optim.Adam(modelG_DtoF.parameters(), lr=args.init_lr)
    optimizerG_FtoD = optim.Adam(modelG_FtoD.parameters(), lr=args.init_lr)
    optimizerD_Full = optim.Adam(modelD_Full.parameters(), lr=args.init_lr)
    optimizerD_Down = optim.Adam(modelD_Down.parameters(), lr=args.init_lr)

    schedulerG_DtoF = optim.lr_scheduler.StepLR(optimizerG_DtoF, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)
    schedulerG_FtoD = optim.lr_scheduler.StepLR(optimizerG_FtoD, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)
    schedulerD_Full = optim.lr_scheduler.StepLR(optimizerD_Full, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)
    schedulerD_Down = optim.lr_scheduler.StepLR(optimizerD_Down, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)

    trainer = ModelTrainerIMGgan(args, modelG_DtoF, modelG_FtoD, modelD_Full, modelD_Down, train_loader,
                                 optimizerG_DtoF, optimizerG_FtoD, optimizerD_Full, optimizerD_Down,
                                 input_train_transform, output_transform, losses, schedulerG_DtoF, schedulerG_FtoD,
                                 schedulerD_Full, schedulerD_Down)

    trainer.train_model()

if __name__ == '__main__':
    settings = dict(
        # Variables that almost never change.
        name='ODT_baseline',  # Please do change this every time Harry
        data_root=None,
        log_root=None,
        ckpt_root=None,
        batch_size=1,
        chans=64,
        num_pool_layers=4,
        save_best_only=False,

        # Variables that occasionally change.
        display_images=40,  # Maximum number of images to save.
        num_workers=3,
        init_lr=1e-4,
        gpu=0,
        max_to_keep=10,
        use_gp=False,
        gan_mode='lsgan',
        clip_limit=0.01,
        use_residual=True,

        start_slice=1,
        start_val_slice=1,

        # Patch processing
        use_patch=True,
        patch_size=320,

        # GAN
        init_GAN_lambda=[1, 3, 5, 7],
        weight_mode='linear',
        GAN_lambda=1e-1,
        GAN_lambda2=1e-1,
        down_scale=4,
        depth=1,
        use_ident=True,
        ident_lambda=0.5,

        # Prev model ckpt
        load_ckpt=False,
        prev_model_ckpt_G_DtoF='./checkpoints/mixres_LSGAN_maskflip_ident/Trial 03  2019-12-15 12-57-53/ckpt_G025.tar',
        prev_model_ckpt_G_FtoD='./checkpoints/mixres_LSGAN_maskflip_ident/Trial 03  2019-12-15 12-57-53/ckpt_G025.tar',
        prev_model_ckpt_D_Full='./checkpoints/mixres_LSGAN_maskflip_ident/Trial 03  2019-12-15 12-57-53/ckpt_D025.tar',
        prev_model_ckpt_D_Down='./checkpoints/mixres_LSGAN_maskflip_ident/Trial 03  2019-12-15 12-57-53/ckpt_D025.tar',
        prev_model_ckpt_D_proj='./checkpoints/mixres_LSGAN_maskflip_ident/Trial 03  2019-12-15 12-57-53/ckpt_D025.tar',

        # Variables that change frequently.
        D_step=3,   # number of discriminator updates per generator update
        sample_rate=1,
        num_epochs=150,
        verbose=False,
        use_slice_metrics=True,  # Using slice metrics causes a 30% increase in training time.
        lr_red_epoch=100,
        lr_red_rate=0.1,
        pool_size=50,

        # Evaluation
        eval_fdir='./test_axial/val_input/',
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)