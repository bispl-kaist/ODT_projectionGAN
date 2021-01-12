import torch
import torch.nn.functional as F
import torchgeometry as tgm

import numpy as np
import random

from data.data_transforms import to_tensor, ifft2, fft2, complex_abs, apply_info_mask, kspace_to_nchw, ifft1, fft1, \
    complex_center_crop, center_crop, root_sum_of_squares, k_slice_to_chw, ej_kslice_to_chw, ej_permute, ej_permute_bchw, \
    extract_patch_transform, extract_patch_transform_proj, extract_patch_transform_single, nchw_to_kspace, \
    extract_patch_transform_inference

class Prefetch2Device:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, input, target, input_file_name, slice_file_name):

        # I hope that async copy works for passing between processes but I am not sure.
        input_slice = to_tensor(input).to(device=self.device)
        target_slice = to_tensor(target).to(device=self.device)

        input_slice = input_slice.unsqueeze(dim=0)
        target_slice = target_slice.unsqueeze(dim=0)

        return input_slice, target_slice, input_file_name, slice_file_name


class Prefetch2Device_double:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, input1, input2, target1, target2, input_file_name1, input_file_name2, target_file_name1, target_file_name2):

        if input1.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')
        if target1.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')

        # I hope that async copy works for passing between processes but I am not sure.
        input_slice1 = to_tensor(input1).to(device=self.device)
        input_slice2 = to_tensor(input2).to(device=self.device)
        target_slice1 = to_tensor(target1).to(device=self.device)
        target_slice2 = to_tensor(target2).to(device=self.device)

        return input_slice1, input_slice2, target_slice1, target_slice2, input_file_name1, input_file_name2, target_file_name1, target_file_name2


class Prefetch2Device_Axial:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, input, target, input_file_name, slice_file_name):

        input_slice = to_tensor(input).to(device=self.device)
        target_slice = to_tensor(target).to(device=self.device)

        input_slice = input_slice.permute(2, 0, 1)
        target_slice = target_slice.permute(2, 0, 1)

        if margin > 0:
            pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
        else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
            pad = [0, 0]

        # This pads at the last dimension of a tensor with 0.
        inputs = F.pad(masked_kspace, pad=pad, value=0)

        return input_slice, target_slice, input_file_name, slice_file_name


class Prefetch2Device_Axial_proj:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, input, target, proj, input_file_name, slice_file_name, proj_file_name):

        if input.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')
        if target.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')

        # I hope that async copy works for passing between processes but I am not sure.
        input_slice = to_tensor(input).to(device=self.device)
        target_slice = to_tensor(target).to(device=self.device)
        proj = to_tensor(proj).to(device=self.device)

        return input_slice, target_slice, proj, input_file_name, slice_file_name, proj_file_name


class Prefetch2DeviceVal:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, target, slice_file_name):
        if target.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')

        # I hope that async copy works for passing between processes but I am not sure.
        target_slice = to_tensor(target).to(device=self.device)

        return target_slice, slice_file_name


class PreProcessScale:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'scales': scale_std}

        return input_slice, target_slice, rss_target, extra_params


class PreProcessScale_nL:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = input_slice * scaling
            target_slice = target_slice
            target_slice_4d = kspace_to_nchw(target_slice)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, extra_params


class PreProcessScale_k:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_k = fft2(input_slice)
            target_k = fft2(target_slice)
            scale_std = torch.std(input_k)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_k = input_k * scaling
            target_k = target_k

            input_k = kspace_to_nchw(input_k)
            target_k = kspace_to_nchw(target_k)

            extra_params = {'input_scales': scale_std}

        return input_k, target_k, extra_params


class PreProcessScale_k_flip:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_k = fft2(input_slice)
            target_k = fft2(target_slice)
            scale_std = torch.std(input_k)
            scaling = 1 / scale_std

            input_k = input_k * scaling
            target_k = target_k

            input_k = kspace_to_nchw(input_k)
            target_k = kspace_to_nchw(target_k)

            flip_choice = random.choice([0, 1, 2, 3])
            if flip_choice == 1:  # Vertical flip
                input_k = torch.flip(input_k, dims=[-2])
                target_k = torch.flip(target_k, dims=[-2])
            elif flip_choice == 2:  # Horizontal flip
                input_k = torch.flip(input_k, dims=[-1])
                target_k = torch.flip(target_k, dims=[-1])
            elif flip_choice == 3:  # Horizontal flip
                input_k = torch.flip(input_k, dims=[-1, -2])
                target_k = torch.flip(target_k, dims=[-1, -2])

            extra_params = {'input_scales': scale_std, 'flip_choice': flip_choice}

        return input_k, target_k, extra_params


class PreProcessScale_nL_flip:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = input_slice * scaling
            target_slice = target_slice
            target_slice_4d = kspace_to_nchw(target_slice)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            flip_choice = random.choice([0, 1, 2, 3])
            if flip_choice == 1:  # Vertical flip
                input_slice = torch.flip(input_slice, dims=[-2])
                target_slice = torch.flip(target_slice, dims=[-2])
            elif flip_choice == 2:  # Horizontal flip
                input_slice = torch.flip(input_slice, dims=[-1])
                target_slice = torch.flip(target_slice, dims=[-1])
            elif flip_choice == 3:  # Horizontal flip
                input_slice = torch.flip(input_slice, dims=[-1, -2])
                target_slice = torch.flip(target_slice, dims=[-1, -2])

            extra_params = {'input_scales': scale_std, 'flip_choice': flip_choice}

        return input_slice, target_slice, extra_params


class PreProcessScale_nL_flip_double:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice1, input_slice2, target_slice1, target_slice2, input_fname1, input_fname2, target_fname1, target_fname2):
        assert isinstance(input_slice1, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice1, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std1 = torch.std(input_slice1)
            scaling1 = 1 / scale_std1
            scale_std2 = torch.std(input_slice2)
            scaling2 = 1 / scale_std2

            input_slice1 = ej_permute_bchw(input_slice1)
            input_slice2 = ej_permute_bchw(input_slice2)
            target_slice1 = ej_permute_bchw(target_slice1)
            target_slice2 = ej_permute_bchw(target_slice2)

            input_slice1 = input_slice1 * scaling1
            input_slice2 = input_slice2 * scaling2

            input_slice1 = kspace_to_nchw(input_slice1)
            input_slice2 = kspace_to_nchw(input_slice2)
            target_slice1 = kspace_to_nchw(target_slice1)
            target_slice2 = kspace_to_nchw(target_slice2)

            flip_choice = random.choice([0, 1, 2, 3])
            if flip_choice == 1:  # Vertical flip
                input_slice1 = torch.flip(input_slice1, dims=[-2])
                input_slice2 = torch.flip(input_slice2, dims=[-2])
                target_slice1 = torch.flip(target_slice1, dims=[-2])
                target_slice2 = torch.flip(target_slice2, dims=[-2])
            elif flip_choice == 2:  # Horizontal flip
                input_slice1 = torch.flip(input_slice1, dims=[-1])
                input_slice2 = torch.flip(input_slice2, dims=[-1])
                target_slice1 = torch.flip(target_slice1, dims=[-1])
                target_slice2 = torch.flip(target_slice2, dims=[-1])
            elif flip_choice == 3:  # Horizontal flip
                input_slice1 = torch.flip(input_slice1, dims=[-1, -2])
                input_slice2 = torch.flip(input_slice2, dims=[-1, -2])
                target_slice1 = torch.flip(target_slice1, dims=[-1, -2])
                target_slice2 = torch.flip(target_slice2, dims=[-1, -2])

            extra_params = {'input_scales1': scale_std1, 'input_scales2': scale_std2, 'flip_choice': flip_choice}

        return input_slice1, input_slice2, target_slice1, target_slice2, extra_params


class PreProcessScale_Axial:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1, use_patch=False, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size
        self.use_patch = use_patch

    def __call__(self, input_slice, target_slice, input_fname, target_fname):

        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling

            if self.use_patch:
                input_slice, target_slice = extract_patch_transform(input_slice, target_slice, patch_size=self.patch_size)

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, extra_params


class PreProcessScale_Rot:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1, use_patch=False, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size
        self.use_patch = use_patch

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std


            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            _, _, h, w = input_slice.shape

            # Rotation with Torch Geometry
            # center
            center = torch.ones(1, 2)
            center[..., 0] = target_slice.shape[-1] / 2
            center[..., 1] = target_slice.shape[-2] / 2

            # angle, scale
            angle = torch.FloatTensor(1).uniform_(0, 360)
            scale = torch.ones(1)
            M = tgm.get_rotation_matrix2d(center, angle, scale)
            M = M.to(self.device)
            input_slice = tgm.warp_affine(input_slice, M, dsize=(h, w))
            target_slice = tgm.warp_affine(target_slice, M, dsize=(h, w))

            if self.use_patch:
                input_slice = extract_patch_transform_single(input_slice, patch_size=self.patch_size)
                target_slice = extract_patch_transform_single(target_slice, patch_size=self.patch_size)

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, extra_params


class PreProcessScale_affine:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1, use_patch=False, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size
        self.use_patch = use_patch

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std


            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            _, _, h, w = input_slice.shape

            # 1. Rotation with Torch Geometry
            # center
            center = torch.ones(1, 2)
            center[..., 0] = target_slice.shape[-1] / 2
            center[..., 1] = target_slice.shape[-2] / 2

            # angle, scale
            angle = torch.FloatTensor(1).uniform_(0, 360)
            scale = torch.ones(1)
            M = tgm.get_rotation_matrix2d(center, angle, scale)
            M = M.to(self.device)
            input_slice = tgm.warp_affine(input_slice, M, dsize=(h, w))
            target_slice = tgm.warp_affine(target_slice, M, dsize=(h, w))

            # 2. (Shearing, Scaling) in general
            Mxx = torch.FloatTensor(1).uniform_(1, 1.5)
            Myy = torch.FloatTensor(1).uniform_(1, 1.5)
            Mxy = torch.FloatTensor(1).uniform_(0, 0.5)
            Myx = torch.FloatTensor(1).uniform_(0, 0.5)

            M = torch.tensor([[[Mxx, Mxy, 0], [Myx, Myy, 0]]]).to(self.device)
            input_slice = tgm.warp_affine(input_slice, M, dsize=(h, w))
            target_slice = tgm.warp_affine(target_slice, M, dsize=(h, w))

            # 3. Flip ud
            rand_ud = random.choice([True, False])
            if rand_ud is True:
                input_slice = torch.flip(input_slice, dims=[-2])
                target_slice = torch.flip(target_slice, dims=[-2])

            # 4. Flip lr
            rand_lr = random.choice([True, False])
            if rand_lr is True:
                input_slice = torch.flip(input_slice, dims=[-1])
                target_slice = torch.flip(target_slice, dims=[-1])

            # 4. Flip udlr
            rand_udlr = random.choice([True, False])
            if rand_udlr is True:
                input_slice = torch.flip(input_slice, dims=[-2, -1])
                target_slice = torch.flip(target_slice, dims=[-2, -1])

            if self.use_patch:
                input_slice = extract_patch_transform_single(input_slice, patch_size=self.patch_size)
                target_slice = extract_patch_transform_single(target_slice, patch_size=self.patch_size)

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, extra_params


class PreProcessScale_Axial_nothing:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, extra_params


class PreProcessScale_A_proj_crop:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, proj_slice, input_fname, target_fname, proj_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'
        assert isinstance(proj_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice.permute(0, 3, 1, 2)
            target_slice = target_slice.permute(0, 3, 1, 2)
            proj_slice = proj_slice.unsqueeze(dim=1)

            input_slice, target_slice = extract_patch_transform(input_slice, target_slice, patch_size=self.patch_size)
            proj_slice = extract_patch_transform_single(proj_slice, patch_size=self.patch_size)

            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            scale_std_proj = torch.std(proj_slice)
            proj_scaling = 1 / scale_std_proj

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            proj_slice = proj_slice * proj_scaling

            extra_params = {'input_scales': scale_std, 'MIP_scales': scale_std_proj}

        return input_slice, target_slice, proj_slice, extra_params


class PreProcessScale_A_proj_crop_v2:
    def __init__(self, device, use_seed=True, divisor=1, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, proj_slice, input_fname, target_fname, proj_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'
        assert isinstance(proj_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice.permute(0, 3, 1, 2)
            target_slice = target_slice.permute(0, 3, 1, 2)
            proj_slice = proj_slice.unsqueeze(dim=1)

            input_slice, target_slice, proj_slice = extract_patch_transform_proj(input_slice, target_slice, proj_slice,
                                                                                 patch_size=self.patch_size)

            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            proj_slice = proj_slice * scaling

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, proj_slice, extra_params


class PreProcessScale_A_proj_crop_v3:
    def __init__(self, device, use_seed=True, divisor=1, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, proj_slice, input_fname, target_fname, proj_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'
        assert isinstance(proj_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice.permute(0, 3, 1, 2)
            target_slice = target_slice.permute(0, 3, 1, 2)
            proj_slice = proj_slice.unsqueeze(dim=1)

            input_slice, target_slice, proj_slice = extract_patch_transform_proj(input_slice, target_slice, proj_slice,
                                                                                 patch_size=self.patch_size)

            proj_slice, _ = torch.max(target_slice, 1)
            proj_slice = proj_slice.unsqueeze(dim=1)
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            proj_slice = proj_slice * scaling

            extra_params = {'input_scales': scale_std}

        return input_slice, target_slice, proj_slice, extra_params


class PreProcessScale_A_proj_crop_prior:
    def __init__(self, device, use_seed=True, divisor=1, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, proj_slice, input_fname, target_fname, proj_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'
        assert isinstance(proj_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice.permute(0, 3, 1, 2)
            target_slice = target_slice.permute(0, 3, 1, 2)
            proj_slice = proj_slice.unsqueeze(dim=1)
            proj_slice, _ = torch.max(target_slice, 1)
            proj_slice = proj_slice.unsqueeze(dim=1)

            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            target_std = torch.std(target_slice)
            target_scaling = 1 / target_std

            input_slice, target_slice, proj_slice = extract_patch_transform_proj(input_slice, target_slice, proj_slice,
                                                                                 patch_size=self.patch_size)

            input_slice = input_slice * scaling
            target_slice = target_slice * target_scaling
            proj_slice = proj_slice * target_scaling

            extra_params = {'input_scales': scale_std, 'target_scales': target_std}

        return input_slice, target_slice, proj_slice, extra_params


    def __init__(self, device, use_seed=True, divisor=1, patch_size=256):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, proj_slice, input_fname, target_fname, proj_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'
        assert isinstance(proj_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice.permute(0, 3, 1, 2)
            target_slice = target_slice.permute(0, 3, 1, 2)
            proj_slice = proj_slice.unsqueeze(dim=1)

            input_slice, target_slice, proj_slice = extract_patch_transform_proj(input_slice, target_slice, proj_slice,
                                                                                 patch_size=self.patch_size)

            proj_slice, _ = torch.max(target_slice, 1)
            proj_slice = proj_slice.unsqueeze(dim=1)

            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            target_std = torch.std(target_slice)
            target_scaling = 1 / target_std

            input_slice = input_slice * scaling
            target_slice = target_slice * target_scaling
            proj_slice = proj_slice * target_scaling

            extra_params = {'input_scales': scale_std, 'target_scales': target_std}

        return input_slice, target_slice, proj_slice, extra_params


class PreProcessScale_nL_flip_tocuda:  # Scale only the D domain inputs / do not scale F domain labels
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        input_slice = input_slice.to(self.device)
        target_slice = target_slice.to(self.device)
        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = input_slice * scaling
            target_slice = target_slice
            target_slice_4d = kspace_to_nchw(target_slice)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            flip_choice = random.choice([0, 1, 2, 3])
            if flip_choice == 1:  # Vertical flip
                input_slice = torch.flip(input_slice, dims=[-2])
                target_slice = torch.flip(target_slice, dims=[-2])
            elif flip_choice == 2:  # Horizontal flip
                input_slice = torch.flip(input_slice, dims=[-1])
                target_slice = torch.flip(target_slice, dims=[-1])
            elif flip_choice == 3:  # Horizontal flip
                input_slice = torch.flip(input_slice, dims=[-1, -2])
                target_slice = torch.flip(target_slice, dims=[-1, -2])

            extra_params = {'input_scales': scale_std, 'flip_choice': flip_choice}

        return input_slice, target_slice, extra_params


class PreProcessCropScale:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice[:, 67:-67, :, :, :]
            target_slice = target_slice[:, 67:-67, :, :, :]

            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'scales': scale_std}

        return input_slice, target_slice, rss_target, extra_params


class PreProcessCropPatchScale:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():

            input_slice = input_slice[:, 84:-84, :, :, :]
            target_slice = target_slice[:, 84:-84, :, :, :]

            input_slice, target_slice = extract_patch_transform(input_slice, target_slice, patch_size=128)

            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = input_slice * scaling
            target_slice = target_slice * scaling
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'scales': scale_std}

        return input_slice, target_slice, rss_target, extra_params


class PreProcessScaleVal:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, target_slice, target_fname):
        assert isinstance(target_slice, torch.Tensor)

        with torch.no_grad():
            scale_std = torch.std(target_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std

            target_slice = ej_permute_bchw(target_slice)

            target_slice = target_slice * scaling
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'scales': scale_std, 'file_name': target_fname}

        return target_slice, rss_target, extra_params


class PreProcessEval:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, target_slice, target_fname):
        assert isinstance(target_slice, torch.Tensor)

        with torch.no_grad():
            target_slice = ej_permute_bchw(target_slice)
            target_slice = target_slice
            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'file_name': target_fname}

        return target_slice, extra_params


class PreProcessInfer:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, input_fname):
        assert isinstance(input_slice, torch.Tensor)

        with torch.no_grad():
            input_slice = ej_permute_bchw(input_slice)
            input_slice = kspace_to_nchw(input_slice)

            extra_params = {'file_name': input_fname}

        return input_slice, extra_params


class PreProcessInfer_A:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, input_fname):
        assert isinstance(input_slice, torch.Tensor)

        with torch.no_grad():
            input_slice = input_slice.permute(0, 3, 1, 2)
            scaling = input_slice.std()
            input_slice = input_slice / scaling

            extra_params = {'file_name': input_fname}

        return input_slice, extra_params


class PreProcessInfer_A_comparison:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, input_fname):
        assert isinstance(input_slice, torch.Tensor)

        with torch.no_grad():
            input_slice = input_slice.unsqueeze(dim=0)
            scaling = input_slice.std()
            input_slice = input_slice / scaling

            extra_params = {'scaling': scaling, 'file_name': input_fname}

        return input_slice, extra_params


class PreProcessInfer_patch:
    def __init__(self, device, use_seed=True, divisor=1, patch_size=128, stride=64):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, input_slice, input_fname):
        assert isinstance(input_slice, torch.Tensor)

        with torch.no_grad():
            input_slice = input_slice.unsqueeze(dim=0)
            scaling = input_slice.std()
            input_slice = input_slice / scaling

            batched_input = extract_patch_transform_inference(input_slice, dims=2, patch_size=self.patch_size, stride=self.stride)

            extra_params = {'scaling': scaling, 'file_name': input_fname}

        return batched_input, extra_params


class PreProcessInfer_3axis:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_axi_slice, input_cor_slice, input_sag_slice, input_axi_fpath, input_cor_fpath, input_sag_fpath):
        assert isinstance(input_sag_slice, torch.Tensor)

        with torch.no_grad():
            input_axi_slice = input_axi_slice.unsqueeze(dim=0).to(device=self.device)
            input_cor_slice = input_cor_slice.unsqueeze(dim=0).to(device=self.device)
            input_sag_slice = input_sag_slice.unsqueeze(dim=0).to(device=self.device)

            axi_scaling = input_axi_slice.std()
            input_axi_slice = input_axi_slice / axi_scaling
            cor_scaling = input_cor_slice.std()
            input_cor_slice = input_cor_slice / cor_scaling
            sag_scaling = input_sag_slice.std()
            input_sag_slice = input_sag_slice / sag_scaling

            extra_params = {'axi_scaling': axi_scaling, 'cor_scaling': cor_scaling, 'sag_scaling': sag_scaling,
                            'axi_file_name': input_axi_fpath, 'cor_file_name': input_cor_fpath, 'sag_file_name': input_sag_fpath}

        return input_axi_slice, input_cor_slice, input_sag_slice, extra_params


class PreProcessScaleVal_nL:  # Do not scale label images
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, target_slice, target_fname):
        assert isinstance(target_slice, torch.Tensor)

        with torch.no_grad():

            target_slice = ej_permute_bchw(target_slice)
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'file_name': target_fname}

        return target_slice, extra_params


class PreProcessScaleVal_k:  # Do not scale label images
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, target_slice, target_fname):
        assert isinstance(target_slice, torch.Tensor)

        with torch.no_grad():
            target_slice = ej_permute_bchw(target_slice)
            target_k = fft2(target_slice)
            target_k = kspace_to_nchw(target_k)

            extra_params = {'file_name': target_fname}

        return target_k, extra_params


class PreProcessScaleVal_nL_tocuda:  # Do not scale label images
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, target_slice, target_fname):
        assert isinstance(target_slice, torch.Tensor)
        with torch.no_grad():

            target_slice = ej_permute_bchw(target_slice)
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'file_name': target_fname}

        return target_slice, extra_params


class PreProcessStandardize:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(target_slice)
            batch_mean = torch.mean(target_slice)
            scaling = 1 / scale_std

            input_slice = ej_permute_bchw(input_slice)
            target_slice = ej_permute_bchw(target_slice)

            input_slice = (input_slice - batch_mean) * scaling
            target_slice = (target_slice - batch_mean) * scaling
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            input_slice = kspace_to_nchw(input_slice)
            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'scales': scale_std, 'mean': batch_mean}

        return input_slice, target_slice, rss_target, extra_params


class PreProcessStandardizeVal:
    def __init__(self, device, use_seed=True, divisor=1):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, target_slice, target_fname):
        assert isinstance(target_slice, torch.Tensor)

        with torch.no_grad():
            scale_std = torch.std(target_slice)
            batch_mean = torch.mean(target_slice)
            scaling = 1 / scale_std

            target_slice = ej_permute_bchw(target_slice)

            target_slice = (target_slice - batch_mean) * scaling
            target_slice_4d = kspace_to_nchw(target_slice)
            rss_target = root_sum_of_squares(target_slice_4d, dim=1)

            target_slice = kspace_to_nchw(target_slice)

            extra_params = {'scales': scale_std, 'mean': batch_mean, 'file_name': target_fname}

        return target_slice, rss_target, extra_params


class PreProcessWK:
    """
    Class for pre-processing weighted k-space.
    However, weighting is optional since a simple function that returns its input can be used to have no weighting.
    """
    def __init__(self, mask_func, weight_func, challenge, device, use_seed=True, divisor=1):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        assert callable(weight_func), '`weight_func` must be a callable function.'
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.weight_func = weight_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, kspace_target, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target, torch.Tensor), 'k-space target was expected to be a Pytorch Tensor.'
        if kspace_target.dim() == 3:  # If the collate function does not expand dimensions for single-coil.
            kspace_target = kspace_target.expand(1, 1, -1, -1, -1)
        elif kspace_target.dim() == 4:  # If the collate function does not expand dimensions for multi-coil.
            kspace_target = kspace_target.expand(1, -1, -1, -1, -1)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        if kspace_target.size(0) != 1:
            raise NotImplementedError('Batch size should be 1 for now.')

        with torch.no_grad():
            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            weighting = self.weight_func(masked_kspace)
            masked_kspace *= weighting

            # img_input is not actually an input but what the input would look like in the image domain.
            img_input = complex_abs(ifft2(masked_kspace))

            # The slope is meaningless as the results always become the same after standardization no matter the slope.
            # The ordering could be changed to allow a difference, but this would make the inputs non-standardized.
            k_scale = torch.std(masked_kspace)
            k_scaling = 1 / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'weightings': weighting}
            extra_params.update(info)
            extra_params.update(attrs)

            # Recall that the Fourier transform is a linear transform.
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target,
                       'img_targets': img_target, 'img_inputs': img_input}

            if kspace_target.size(1) == 15:
                rss_target = target * k_scaling
                targets['rss_targets'] = rss_target  # rss_target is in 2D

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params