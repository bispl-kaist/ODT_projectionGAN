import torch
from torch import nn, optim
from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_cycle_data_loaders_phantom_axial_infer

from data.input_transforms import PreProcessInfer_patch
from data.output_transforms import OutputTransformPatch

from models.fc_unet import Unet, UnetSA_light
from eval_scripts.Inference_phantom_patch import Infer_CU


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

    # UNET architecture requires that all inputs be dividable by some power of 2.
    divisor = 2 ** args.num_pool_layers

    eval_input_transform = PreProcessInfer_patch(args.device, use_seed=False, divisor=divisor, patch_size=args.patch_size, stride=args.stride)

    # DataLoaders
    eval_loader = create_cycle_data_loaders_phantom_axial_infer(args)

    eval_output_transform = OutputTransformPatch(patch_size=args.patch_size, stride=args.stride)

    data_chans = args.depth

    modelG = Unet(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
                  num_pool_layers=args.num_pool_layers).to(device)

    if args.load_ckpt:
        model_ckpt = args.prev_model_ckpt
        save_dict = torch.load(model_ckpt)
        modelG.load_state_dict(save_dict['model_state_dict'])
        print('Loaded model checkpoint')

    trainer = Infer_CU(args, modelG, eval_loader, eval_input_transform, eval_output_transform)

    trainer.inference_axial_from_coronal(args)


if __name__ == '__main__':

    settings = dict(
        # Variables that almost never change.
        name='test',
        data_root='/media/harry/mri1/backup/ODT/data/TomoGAN_db_infer_multiball',
        specimen_num='all',
        log_root='./logs/test/',
        ckpt_root='./checkpoints/test/',
        batch_size=1,
        chans=32,
        num_pool_layers=3,

        # Variables that occasionally change.
        num_workers=0,
        gpu=0,  # Set to None for CPU mode.
        use_residual=False,
        depth=1,
        verbose=False,
        sample_rate=1,

        patch_size=192,
        stride=24,

        # Prev model ckpt
        load_ckpt=True,
        prev_model_ckpt='./checkpoints/phantom/ckpt.tar',

        # Evaluation
        save_fdir='./recons/phantom'
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)