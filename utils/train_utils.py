import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pathlib import Path

from data.mri_data import SliceData, CustomSliceData, TOFData, TOFData_val, TOFData_CtoA, TOFDataAxial, TOFData_inference, \
TOFData_v2, TOFData_val_v2, TOFData_Axial_Proj, TOFData_inference_A, TOFData_Axial, TOFData_cs4, TOFData_cs8, TOFData_inference_all, \
TOFData_inference_all_cs4, TOFData_inference_all_cs8, TOFData_inference_A_all, TOFData_cs8_0504, TOFData_double, TOFData_multislice, \
SAIT_cycleGAN_data, SAIT_cycleGAN_data_infer, ODT_data, ODT_data_infer, ODT_data_infer_3axis, ODT_data_infer_3axis_microbead, \
ODT_data_infer_3axis_phantom, ODT_data_phantom_infer, AAPM_infer, AAPM
from data.data_transforms import complex_abs, ifft2, normalize_im
from data.input_transforms import Prefetch2Device, Prefetch2DeviceVal, Prefetch2Device_Axial_proj, Prefetch2Device_Axial, \
Prefetch2Device_double

from PIL import Image
import numpy as np
import random


class CheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, model, optimizer, mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(model, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizer, optim.Optimizer), 'Not a Pytorch Optimizer'
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        assert mode in ('min', 'max'), 'Mode must be either `min` or `max`'
        ckpt_path = Path(ckpt_dir)
        assert ckpt_path.exists(), 'Not a valid, existing path'

        record_path = ckpt_path / 'Checkpoints.txt'

        try:
            record_file = open(record_path, mode='x')
        except FileExistsError:
            import sys
            print('WARNING: It is recommended to have a separate checkpoint directory for each run.', file=sys.stderr)
            print('Appending to previous Checkpoint record file!', file=sys.stderr)
            record_file = open(record_path, mode='a')

        print(f'Checkpoint List for {ckpt_path}', file=record_file)
        record_file.close()

        self.model = model
        self.optimizer = optimizer
        self.save_best_only = save_best_only
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.save_counter = 0
        self.record_path = record_path
        self.record_dict = dict()

        if mode == 'min':
            self.prev_best = float('inf')
            self.mode = mode
        elif mode == 'max':
            self.prev_best = -float('inf')
            self.mode = mode
        else:
            raise TypeError('Mode must be either `min` or `max`')

    def _save(self, ckpt_name=None, **save_kwargs):
        self.save_counter += 1
        save_dict = {'model_state_dict': self.model.state_dict()}
        save_dict.update(save_kwargs)
        save_path = self.ckpt_path / (f'{ckpt_name}.tar' if ckpt_name else f'ckpt_{self.save_counter:03d}.tar')

        torch.save(save_dict, save_path)
        print(f'Saved Checkpoint to {save_path}')
        print(f'Checkpoint {self.save_counter:04d}: {save_path}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_path}', file=file)

        self.record_dict[self.save_counter] = save_path

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_path

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_path = None
        if is_best or not self.save_best_only:
            save_path = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_path, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dir, load_optimizer=True):
        save_dict = torch.load(load_dir)

        self.model.load_state_dict(save_dict['model_state_dict'])
        print(f'Loaded model parameters from {load_dir}')

        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dir}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


def load_model_from_checkpoint(model, load_dir, strict=False):
    """
    A simple function for loading checkpoints without having to use Checkpoint Manager. Very useful for evaluation.
    Checkpoint manager was designed for loading checkpoints before resuming training.

    model (nn.Module): Model architecture to be used.
    load_dir (str): File path to the checkpoint file. Can also be a Path instead of a string.
    """
    assert isinstance(model, nn.Module), 'Model must be a Pytorch module.'
    assert Path(load_dir).exists(), 'The specified directory does not exist'
    save_dict = torch.load(load_dir)
    model.load_state_dict(save_dict['model_state_dict'], strict=strict)
    return model  # Not actually necessary to return the model but doing so anyway.


def create_datasets(args, train_transform, val_transform):
    assert callable(train_transform) and callable(val_transform), 'Transforms should be callable functions.'

    # Generating Datasets.
    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=train_transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=val_transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False
    )
    return train_dataset, val_dataset


def create_inference_datasets(args, transform):

    # val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    dataset = TOFData_inference(
        # root=Path(args.data_root) / 'train_new/Input',
        root=Path(args.data_root) / 'test/Input',
        patient=args.patient,
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_inference_datasets_all(args, transform):

    # val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    dataset = TOFData_inference_all(
        root=Path(args.data_root) / 'test',
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_inference_datasets_traindb_cs4_all(args, transform):

    # val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    dataset = TOFData_inference_all_cs4(
        root=Path(args.data_root) / 'train_new/Input',
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_inference_datasets_traindb_cs8_all(args, transform):

    # val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    dataset = TOFData_inference_all_cs8(
        # root=Path(args.data_root) / 'train_new/Input',
        root=Path(args.data_root) / 'train_new_vol/Input',
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_inference_datasets_A(args, transform):

    # Generating Datasets.
    dataset = TOFData_inference_A(
        root=Path(args.data_root) / 'test_axial/Input',
        patient=args.patient,
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_inference_datasets_A_all(args, transform):

    # Generating Datasets.
    dataset = TOFData_inference_A_all(
        # root=Path(args.data_root) / 'train_axial_2.5D_7/Input',
        # root=Path(args.data_root) / 'train_axial_2.5D_CS8_7_0519/Input',
        root=Path(args.data_root) / 'train_axial_2.5D_CS8_7_0530/Input',
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_inference_datasets_A_comparison(args, transform):

    # Generating Datasets.
    dataset = TOFData_inference_A(
        root=Path(args.data_root) / 'test_axial_retro_2.5D_CS8/Input',
        patient=args.patient,
        transform=None,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_CtoA_datasets(args, transform):

    # val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    dataset = TOFData_CtoA(
        root=Path(args.data_root) / 'val/Label',
        patient=args.patient,
        transform=None,
        # transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return dataset


def create_CtoA_datasets_val(args, transform):

    # val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    dataset = TOFData_CtoA(
        root=Path(args.data_root) / 'val/Label',
        # root=Path(args.data_root),
        patient=args.patient,
        transform=None,
        # transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return dataset


def single_collate_fn(batch):  # Returns `targets` as a 4D Tensor.
    """
    hack for single batch case.
    """
    return batch[0][0].unsqueeze(0), batch[0][1].unsqueeze(0), batch[0][2]


def single_triplet_collate_fn(batch):
    return batch[0][0], batch[0][1], batch[0][2]


def single_batch_collate_fn(batch):
    return batch[0]


def multi_collate_fn(data):

    kspace_target, target, attrs, file_name, slice_num = zip(*data)

    max_width = 0
    for tensor in kspace_target:
        if max_width <= tensor.size(-2):
            max_width = tensor.size(-2)

    for idx, tensor in enumerate(kspace_target):
        # If width within batch does not match
        if max_width > tensor.size(-2):
            pad = [0, 0, (max_width - tensor.size(-2)) // 2, (max_width - tensor.size(-2)) // 2]
        else:
            pad = [0, 0]
        # print(f'before pad tensor shape: {tensor.shape}')
        tensor = F.pad(tensor, pad=pad, value=0)
        if idx == 0:
            k_target = tensor.unsqueeze(dim=0)
        else:
            k_target = torch.cat((k_target, tensor.unsqueeze(dim=0)), dim=0)
        # print(f'k_target shape: {k_target.shape}')

    return k_target, target, attrs, file_name, slice_num


def create_data_loaders(args, train_transform, val_transform):

    """
    A function for creating datasets where the data is sent to the desired device before being given to the model.
    This is done because data transfer is a serious bottleneck in k-space learning and is best done asynchronously.
    Also, the Fourier Transform is best done on the GPU instead of on CPU.
    Finally, Sending k-space data to device beforehand removes the need to also send generated label data to device.
    This reduces data transfer significantly.
    The only problem is that sending to GPU cannot be batched with this method.
    However, this seems to be a small price to pay.
    """
    assert callable(train_transform) and callable(val_transform), 'Transforms should be callable functions.'

    train_dataset, val_dataset = create_datasets(args, train_transform, val_transform)

    if args.batch_size == 1:
        collate_fn = single_collate_fn
    elif args.batch_size > 1:
        collate_fn = multi_collate_fn
    else:
        raise RuntimeError('Invalid batch size')

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )
    return train_loader, val_loader


def create_custom_datasets(args, transform=None):

    transform = Prefetch2Device(device=args.device) if transform is None else transform

    # Generating Datasets.
    train_dataset = CustomSliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        start_slice=args.start_slice,
        use_gt=True,
    )

    val_dataset = CustomSliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        start_slice=args.start_val_slice,
        use_gt=True,
    )

    return train_dataset, val_dataset


def create_cycle_datasets(args, transform=None):

    train_transform = Prefetch2Device(device=args.device)
    val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData(
        root=Path(args.data_root) / 'train_new',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val(
        root=Path(args.data_root) / 'val',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cycle_datasets_double(args, transform=None):

    train_transform = Prefetch2Device_double(device=args.device)
    val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_double(
        root=Path(args.data_root) / 'train_new',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val(
        root=Path(args.data_root) / 'val',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cycle_datasets_cs8(args, transform=None):

    train_transform = Prefetch2Device(device=args.device)
    val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_cs8(
        root=Path(args.data_root) / 'train_new',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val(
        root=Path(args.data_root) / 'val',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cycle_datasets_cs8_0504(args, transform=None):

    train_transform = Prefetch2Device(device=args.device)
    val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_cs8_0504(
        root=Path(args.data_root) / 'train_new',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val(
        root=Path(args.data_root) / 'val',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cycle_datasets_multislice(args, transform=None):

    train_transform = Prefetch2Device(device=args.device)
    val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_multislice(
        root=Path(args.data_root) / 'train_new_vol',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val(
        root=Path(args.data_root) / 'val_vol',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cycle_datasets_cs4(args, transform=None):

    train_transform = Prefetch2Device(device=args.device)
    val_transform = Prefetch2DeviceVal(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_cs4(
        root=Path(args.data_root) / 'train_new',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val(
        root=Path(args.data_root) / 'val',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cycle_datasets_axial_cs4(args, transform=None):

    train_transform = Prefetch2Device_Axial_proj(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_Axial_Proj(
        root=Path(args.data_root) / 'train_axial_2.5D_7',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset


def create_cycle_datasets_axial_cs8(args, transform=None):

    train_transform = Prefetch2Device_Axial_proj(device=args.device)

    # Generating Datasets.
    train_dataset = TOFData_Axial_Proj(
        root=Path(args.data_root) / 'train_axial_2.5D_CS8_7_0530',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset


def create_cycle_datasets_axial_single(args, transform=None):

    train_transform = Prefetch2Device_Axial(device=args.device)

    # Generating Datasets.
    train_dataset = ODT_data(
        root=Path(args.data_root),
        transform=transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset


def create_AAPM_cGAN_datasets(args, transform=None):

    train_transform = Prefetch2Device_Axial(device=args.device)
    # Generating Datasets.
    train_dataset = AAPM(
        root=Path(args.data_root) / 'train',
        # root=Path(args.data_root),
        transform=transform,
    )

    # val_dataset = AAPM(
    #     root=Path(args.data_root) / 'val',
    #     # root=Path(args.data_root),
    #     transform=transform,
    # )

    return train_dataset


def create_cycle_datasets_axial_infer(args, transform=None):

    # Generating Datasets.
    test_dataset = ODT_data_infer(
        root=Path(args.data_root),
        transform=transform,
        specimen_type=args.specimen_type,
        specimen_fname=args.specimen_fname,
    )

    return test_dataset


def create_cycle_datasets_phantom_axial_infer(args, transform=None):

    # ODT phantom
    test_dataset = ODT_data_phantom_infer(
        root=Path(args.data_root),
        transform=transform,
        specimen_num=args.specimen_num,
    )

    return test_dataset


def create_cycle_datasets_axial_infer_3axis(args, transform=None):

    # Generating Datasets.
    test_dataset = ODT_data_infer_3axis(
        root=Path(args.data_root),
        transform=transform,
        specimen_type=args.specimen_type,
        specimen_fname=args.specimen_fname,
    )

    return test_dataset


def create_cycle_datasets_axial_infer_3axis_microbead(args, transform=None):

    # Generating Datasets.
    test_dataset = ODT_data_infer_3axis_microbead(
        root=Path(args.data_root),
        transform=transform,
        specimen_type=args.specimen_type,
    )

    return test_dataset

def create_cycle_datasets_axial_infer_3axis_phantom(args, transform=None):

    # Generating Datasets.
    test_dataset = ODT_data_infer_3axis_phantom(
        root=Path(args.data_root),
        transform=transform,
    )

    return test_dataset


def create_cycle_datasets_v2(args, transform=None):

    # Generating Datasets.
    train_dataset = TOFData_v2(
        root=Path(args.data_root) / 'train_0111',
        transform=None,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFData_val_v2(
        root=Path(args.data_root) / 'val',
        transform=None,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_cgan_datasets(args, transform=None):

    train_transform = Prefetch2Device(device=args.device)
    val_transform = Prefetch2Device(device=args.device)

    # Generating Datasets.
    train_dataset = TOFDataAxial(
        root=Path(args.data_root) / 'train',
        transform=train_transform,
        sample_rate=args.sample_rate,
    )

    val_dataset = TOFDataAxial(
        root=Path(args.data_root) / 'val',
        transform=val_transform,
        sample_rate=args.sample_rate,
    )

    return train_dataset, val_dataset


def create_custom_data_loaders(args, transform=None):
    train_dataset, val_dataset = create_custom_datasets(args, transform)

    # collate_fn = single_batch_collate_fn

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        # collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        # collate_fn=collate_fn
    )
    return train_loader, val_loader


def create_cycle_data_loaders(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cycle_data_loaders_double(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets_double(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cycle_data_loaders_cs4(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets_cs4(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cycle_data_loaders_cs8(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets_cs8(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cycle_data_loaders_cs8_0504(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets_cs8_0504(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cycle_data_loaders_multislice(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets_multislice(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cycle_data_loaders_axial(args, type='cs8', transform=None):
    if type == 'cs4':
        train_dataset = create_cycle_datasets_axial_cs4(args, transform)
    elif type == 'cs8':
        train_dataset = create_cycle_datasets_axial_cs8(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_cycle_data_loaders_axial_single(args, transform=None):
    train_dataset = create_cycle_datasets_axial_single(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_AAPM_cGAN_data_loaders(args, transform=None):
    # train_dataset, val_dataset = create_AAPM_cGAN_datasets(args, transform)
    train_dataset = create_AAPM_cGAN_datasets(args, transform)
    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    # )

    return train_loader


def create_cycle_data_loaders_axial_infer(args, transform=None):
    train_dataset = create_cycle_datasets_axial_infer(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_cycle_data_loaders_phantom_axial_infer(args, transform=None):
    train_dataset = create_cycle_datasets_phantom_axial_infer(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_cycle_data_loaders_axial_infer_3axis(args, transform=None):
    train_dataset = create_cycle_datasets_axial_infer_3axis(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_cycle_data_loaders_axial_infer_3axis_microbead(args, transform=None):
    train_dataset = create_cycle_datasets_axial_infer_3axis_microbead(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_cycle_data_loaders_axial_infer_3axis_phantom(args, transform=None):
    train_dataset = create_cycle_datasets_axial_infer_3axis_phantom(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader


def create_cycle_data_loaders_v2(args, transform=None):
    train_dataset, val_dataset = create_cycle_datasets_v2(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_cgan_data_loaders(args, transform=None):

    train_dataset, val_dataset = create_cgan_datasets(args, transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_inference_data_loaders(args, transform=None):
    dataset = create_inference_datasets(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_all(args, transform=None):
    dataset = create_inference_datasets_all(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_traindb_cs4_all(args, transform=None):
    dataset = create_inference_datasets_traindb_cs4_all(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_traindb_cs8_all(args, transform=None):
    dataset = create_inference_datasets_traindb_cs8_all(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_A(args, transform=None):
    dataset = create_inference_datasets_A(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_A_all(args, transform=None):
    dataset = create_inference_datasets_A_all(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_A_comparison(args, transform=None):
    dataset = create_inference_datasets_A_comparison(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_inference_data_loaders_A_single(args, transform=None):
    dataset = create_inference_datasets_A(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_CtoA_data_loaders(args, transform=None):
    dataset = create_CtoA_datasets(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def create_CtoA_val_data_loaders(args, transform=None):
    dataset = create_CtoA_datasets_val(args, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return loader


def make_grid_doublet(image_inputs, image_recons):
    assert image_inputs.size() == image_recons.size()
    irt = torch.cat((image_inputs, image_recons), dim=1)
    return irt


def make_grid_triplet(image_inputs, image_recons, image_targets):
    assert image_inputs.size() == image_recons.size() == image_targets.size()
    irt = torch.cat((image_inputs, image_recons, image_targets), dim=1)
    return irt


def make_RSS(image_recons, image_targets):

    image_recons = image_recons[0].detach().squeeze()
    image_targets = image_targets[0].detach()
    assert image_recons.size() == image_targets.size()

    large = torch.max(image_targets)
    small = torch.min(image_targets)
    diff = large - small

    # Scale again
    image_recons = (image_recons.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)
    image_targets = (image_targets - small) * (torch.tensor(1) / diff)

    image_recons = image_recons.squeeze().cpu()
    image_targets = image_targets.squeeze().cpu()

    deltas = torch.abs(image_targets - image_recons)

    large_delta = torch.max(deltas)
    small_delta = torch.min(deltas)
    diff_delta = large_delta - small_delta

    deltas = (deltas - small_delta) * (torch.tensor(1) / diff_delta)

    return image_recons, image_targets, deltas


def make_recons(image_recons):

    image_recons = image_recons[0].detach().squeeze()

    large = torch.max(image_recons)
    small = torch.min(image_recons)
    diff = large - small

    # Scale again
    image_recons = (image_recons.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)
    image_recons = image_recons.squeeze().cpu()

    return image_recons


def make_normalize(image_recons, image_targets):
    large = torch.max(image_targets)
    small = torch.min(image_targets)
    diff = large - small

    image_recons = (image_recons.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)
    image_targets = (image_targets - small) * (torch.tensor(1) / diff)

    deltas = image_targets - image_recons

    return image_recons, image_targets, deltas


def make_input_RSS(image_inputs):

    image_inputs = image_inputs[0].detach().squeeze()

    # RSS
    image_inputs = (image_inputs ** 2).sum(dim=0).sqrt()

    large = torch.max(image_inputs)
    small = torch.min(image_inputs)
    diff = large - small

    image_inputs = (image_inputs.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)

    image_inputs = image_inputs.squeeze().cpu()

    return image_inputs


def make_input_triplet(image_inputs):

    large = torch.max(image_inputs)
    small = torch.min(image_inputs)
    diff = large - small

    image_inputs = (image_inputs.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)
    image_inputs = image_inputs.detach().squeeze(dim=0)
    image_inputs = torch.cat(torch.chunk(image_inputs.view(-1, image_inputs.size(-1)), chunks=5, dim=0), dim=1)
    image_inputs = image_inputs.squeeze().cpu()

    return image_inputs


def make_k_grid(kspace_recons, smoothing_factor=8):
    """
    Function for making k-space visualizations for Tensorboard.
    """
    # Simple hack. Just use the first element if the input is a list --> batching implementation.
    if isinstance(kspace_recons, list):
        kspace_recons = kspace_recons[0].unsqueeze(dim=0)

    # if kspace_recons.size(0) > 1:
    #     raise NotImplementedError('Mini-batch size greater than 1 has not been implemented yet.')

    # Assumes that the smallest values will be close enough to 0 as to not matter much.
    kspace_recons = kspace_recons[0]
    kspace_view = complex_abs(kspace_recons.detach()).squeeze(dim=0)
    # Scaling & smoothing.
    # smoothing_factor converted to float32 tensor. expm1 and log1p require float32 tensors.
    # They cannot accept python integers.
    sf = torch.tensor(smoothing_factor, dtype=torch.float32)
    kspace_view *= torch.expm1(sf) / kspace_view.max()
    kspace_view = torch.log1p(kspace_view)  # Adds 1 to input for natural log.
    kspace_view /= kspace_view.max()  # Normalization to 0~1 range.

    if kspace_view.size(0) == 15:
        kspace_view = torch.cat(torch.chunk(kspace_view.view(-1, kspace_view.size(-1)), chunks=5, dim=0), dim=1)

    return kspace_view.squeeze().cpu()


def visualize_from_kspace(kspace_recons, kspace_targets, smoothing_factor=4):
    """
    Assumes that all values are on the same scale and have the same shape.
    """
    image_recons = complex_abs(ifft2(kspace_recons))
    image_targets = complex_abs(ifft2(kspace_targets))
    image_recons, image_targets, image_deltas = make_grid_triplet(image_recons, image_targets)
    kspace_targets = make_k_grid(kspace_targets, smoothing_factor)
    kspace_recons = make_k_grid(kspace_recons, smoothing_factor)
    return kspace_recons, kspace_targets, image_recons, image_targets, image_deltas


def imsave(tensor, fname):
    assert isinstance(tensor, torch.Tensor), "input should be a pytorch tensor"
    assert isinstance(fname, str), "file name should be given as string"
    normalized_tensor = normalize_im(tensor)
    np_normalized_tensor = normalized_tensor.squeeze().cpu().numpy()
    np_normalized_tensor_int = (np_normalized_tensor * 255).astype(np.uint8)
    visual_im = Image.fromarray(np_normalized_tensor_int)
    visual_im.save(fname)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def get_scale_weights(i, total_epochs, init_lambda, mode='linear'):
    '''
    Outputs tensor of length 'scale',
    len(init_lambda) == scale
    '''
    if mode == 'linear':
        init_lambda_rev, _ = torch.sort(init_lambda, descending=True)
        diff = (init_lambda_rev - init_lambda) * (i / total_epochs)
    elif mode == 'vanilla':
        diff = torch.zeros_like(init_lambda)

    return init_lambda + diff


def stratify_images(img1, img2, level):
    '''
    Stratifies the given 2 images into certain levels
    Assumes single batch
    '''
    assert img1.shape == img2.shape, \
        f'The two images taken as input have different sizes - img1: {img1.shape} / img2: {img2.shape}'

    rep_img1 = img1.repeat(1, level, 1, 1)
    rep_img2 = img2.repeat(1, level, 1, 1)

    max1 = torch.max(img1)
    max2 = torch.max(img2)

    th1 = (max1 / level)
    th2 = (max2 / level)
    for i in range(0, level):
        thmin1 = (i * th1)
        thmin2 = (i * th2)
        rep_img1_slice = rep_img1[0, i, :, :]
        rep_img2_slice = rep_img2[0, i, :, :]
        rep_img1_slice[rep_img1_slice < thmin1] = 0
        rep_img2_slice[rep_img2_slice < thmin2] = 0

    return rep_img1, rep_img2


