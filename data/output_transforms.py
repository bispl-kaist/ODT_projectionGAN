import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from data.data_transforms import nchw_to_kspace, ifft2, fft2, complex_abs, ifft1, fft1, \
    root_sum_of_squares, center_crop, complex_center_crop, extract_patch_trasnform_inference_gather


class SingleOutputTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, extra_params):
        return outputs * extra_params['scaling']


class OutputTransformPatch(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, batched_outputs, FOV, extra_params):
        recon = extract_patch_trasnform_inference_gather(batched_outputs, FOV, dims=2, patch_size=self.patch_size, stride=self.stride)
        return recon * extra_params['scaling']


class OutputTransform_3axis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, axi_outputs, cor_outputs, sag_outputs, extra_params):
        axi_outputs = axi_outputs * extra_params['axi_scaling']
        axi_outputs = axi_outputs.squeeze()
        cor_outputs = cor_outputs * extra_params['cor_scaling']
        cor_outputs = cor_outputs.squeeze()
        sag_outputs = sag_outputs * extra_params['sag_scaling']
        sag_outputs = sag_outputs.squeeze()
        return axi_outputs, cor_outputs, sag_outputs


class OutputTransform_multislice(nn.Module):
    '''
        Input transform used for 2.5D training with 3 slice input / 3 slice output
        Reconstruction has 180 channels, and this input transform divides it into 3-60channel slices
        Also, each slices are then rssq-ed, which is returned as output
        Maxpooled rssq images are also returned as output
    '''
    def __init__(self, slice, device):
        super().__init__()
        self.slice = slice
        self.device = device

    def forward(self, outputs, extra_params):
        # Split the data into single slices
        ch = outputs.shape[1] // self.slice
        split_outputs = torch.split(outputs, 60, dim=1) # returns a list of 3 tensors
        rss_outputs_list = list()
        for i in range(self.slice):
            rss_outputs = root_sum_of_squares(split_outputs[i], dim=1).unsqueeze(dim=1)
            rss_outputs_list.append(rss_outputs)
        rss_outputs_stack = torch.cat(rss_outputs_list, dim=1)
        mip_outputs = torch.max(rss_outputs_stack, dim=1)

        return outputs, rss_outputs_stack, mip_outputs.values


class DoubleOutputTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs1, outputs2, extra_params):
        return outputs1, outputs2


class OutputInputScaleTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, extra_params):
        return outputs * extra_params['input_scales']


class OutputScaleTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, extra_params):
        return outputs * extra_params['scales']


class OutputEvalTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, scales):
        outputs = outputs * scales
        rss_outputs = root_sum_of_squares(outputs, dim=1)

        return outputs, rss_outputs


class OutputStandardizeTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, extra_params):
        recon = outputs * extra_params['scales'] + extra_params['mean']
        target = targets * extra_params['scales'] + extra_params['mean']
        return recon, target


class WeightedReplacePostProcessK(nn.Module):
    def __init__(self, weighted=True, replace=True):
        super().__init__()
        self.weighted = weighted
        self.replace = replace

    def forward(self, kspace_outputs, targets, extra_params):
        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one slice at a time for now.')

        kspace_targets = targets['kspace_targets']

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - kspace_targets.size(-2)) // 2
        right = left + kspace_targets.size(-2)

        # Cropping width dimension by pad.
        kspace_recons = nchw_to_kspace(kspace_outputs[..., left:right])

        assert kspace_recons.shape == kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (kspace_recons.size(-3) % 2 == 0) and (kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            kspace_recons = kspace_recons / weighting

        if self.replace:  # Replace with original k-space if replace=True
            mask = extra_params['masks']
            kspace_recons = kspace_recons * (1 - mask) + kspace_targets * mask

        cmg_recons = ifft2(kspace_recons)
        img_recons = complex_abs(cmg_recons)
        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class WeightedReplacePostProcessSemiK(nn.Module):
    def __init__(self, weighted=True, replace=True, direction='height'):
        super().__init__()
        self.weighted = weighted
        self.replace = replace

        if direction == 'height':
            self.recon_direction = 'width'
        elif direction == 'width':
            self.recon_direction = 'height'
        else:
            raise ValueError('`direction` should either be `height` or `width')

        self.direction = direction

    def forward(self, semi_kspace_outputs, targets, extra_params):
        if semi_kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        semi_kspace_targets = targets['semi_kspace_targets']
        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (semi_kspace_outputs.size(-1) - semi_kspace_targets.size(-2)) // 2
        right = left + semi_kspace_targets.size(-2)

        # Cropping width dimension by pad.
        semi_kspace_recons = nchw_to_kspace(semi_kspace_outputs[..., left:right])

        assert semi_kspace_recons.shape == semi_kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (semi_kspace_recons.size(-3) % 2 == 0) and (semi_kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            semi_kspace_recons = semi_kspace_recons / weighting

        if self.replace:
            mask = extra_params['masks']
            semi_kspace_recons = semi_kspace_recons * (1 - mask) + semi_kspace_targets * mask

        kspace_recons = fft1(semi_kspace_recons, direction=self.direction)
        cmg_recons = ifft1(semi_kspace_recons, direction=self.recon_direction)
        img_recons = complex_abs(cmg_recons)

        recons = {'semi_kspace_recons': semi_kspace_recons, 'kspace_recons': kspace_recons,
                  'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class PostProcessWK(nn.Module):
    def __init__(self, weighted=True, replace=True, residual_acs=False, resolution=320):
        super().__init__()
        self.weighted = weighted
        self.replace = replace
        self.resolution = resolution
        self.residual_acs = residual_acs

    def forward(self, kspace_outputs, targets, extra_params):
        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one slice at a time for now.')

        kspace_targets = targets['kspace_targets']

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - kspace_targets.size(-2)) // 2
        right = left + kspace_targets.size(-2)

        # Cropping width dimension by pad.
        kspace_recons = nchw_to_kspace(kspace_outputs[..., left:right])
        assert kspace_recons.shape == kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (kspace_recons.size(-3) % 2 == 0) and (kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            kspace_recons = kspace_recons / weighting

        if self.residual_acs:
            num_low_freqs = extra_params['num_low_frequency']
            acs_mask = find_acs_mask(kspace_recons, num_low_freqs)
            kspace_recons = kspace_recons + acs_mask * kspace_targets

        if self.replace:  # Replace with original k-space if replace=True
            mask = extra_params['masks']
            kspace_recons = kspace_recons * (1 - mask) + kspace_targets * mask

        cmg_recons = ifft2(kspace_recons)
        img_recons = complex_abs(cmg_recons)
        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        if img_recons.size(1) == 15:
            top = (img_recons.size(-2) - self.resolution) // 2
            left = (img_recons.size(-1) - self.resolution) // 2
            rss_recon = img_recons[:, :, top:top + self.resolution, left:left + self.resolution]
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()  # rss_recon is in 2D
            recons['rss_recons'] = rss_recon

        return recons  # Returning scaled reconstructions. Not rescaled.


class PostProcessWSemiK(nn.Module):
    def __init__(self, challenge, weighted=True, replace=True, residual_acs=False, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.weighted = weighted
        self.replace = replace
        self.resolution = resolution
        self.residual_acs = residual_acs

    def forward(self, semi_kspace_outputs, targets, extra_params):
        if semi_kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        semi_kspace_targets = targets['semi_kspace_targets']
        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (semi_kspace_outputs.size(-1) - semi_kspace_targets.size(-2)) // 2
        right = left + semi_kspace_targets.size(-2)

        # Cropping width dimension by pad.
        semi_kspace_recons = nchw_to_kspace(semi_kspace_outputs[..., left:right])

        assert semi_kspace_recons.shape == semi_kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (semi_kspace_recons.size(-3) % 2 == 0) and (semi_kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            semi_kspace_recons = semi_kspace_recons / weighting

        if self.residual_acs:
            num_low_freqs = extra_params['num_low_frequency']
            acs_mask = find_acs_mask(semi_kspace_recons, num_low_freqs)
            semi_kspace_recons = semi_kspace_recons + acs_mask * semi_kspace_targets

        if self.replace:
            mask = extra_params['masks']
            semi_kspace_recons = semi_kspace_recons * (1 - mask) + semi_kspace_targets * mask

        kspace_recons = fft1(semi_kspace_recons, direction='height')
        cmg_recons = ifft1(semi_kspace_recons, direction='width')
        img_recons = complex_abs(cmg_recons)

        recons = {'semi_kspace_recons': semi_kspace_recons, 'kspace_recons': kspace_recons,
                  'cmg_recons': cmg_recons, 'img_recons': img_recons}

        if self.challenge == 'multicoil':
            rss_recons = center_crop(img_recons, (self.resolution, self.resolution))
            rss_recons = root_sum_of_squares(rss_recons, dim=1).squeeze()
            rss_recons *= extra_params['sk_scales']  # This value was divided in the inputs. It is thus multiplied here.
            recons['rss_recons'] = rss_recons

        return recons  # Returning scaled reconstructions. Not rescaled. RSS images are rescaled.


def find_acs_mask(kspace_recons: torch.Tensor, num_low_freqs: int):
    assert kspace_recons.dim() == 5, 'Reconstructed tensor in k-space format is expected.'
    num_cols = kspace_recons.size(-2)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask = np.zeros(num_cols, dtype=bool)
    mask[pad:pad+num_low_freqs] = True
    mask = torch.from_numpy(mask).to(dtype=kspace_recons.dtype, device=kspace_recons.device).view(1, 1, 1, -1, 1)
    return mask


class PostProcessCMG(nn.Module):

    def __init__(self, challenge, residual_acs=False, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.residual_acs = residual_acs
        self.resolution = resolution

    def forward(self, cmg_output, targets, extra_params):
        if cmg_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        cmg_target = targets['cmg_targets']
        cmg_recon = nchw_to_kspace(cmg_output)
        assert cmg_recon.shape == cmg_target.shape, 'Reconstruction and target sizes are different.'
        assert (cmg_recon.size(-3) % 2 == 0) and (cmg_recon.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        if self.residual_acs:  # Adding the semi-k-space of the ACS as a residual. Necessary due to complex cropping.
            raise NotImplementedError('Not ready yet.')
            # cmg_acs = targets['cmg_acss']
            # cmg_recon = cmg_recon + cmg_acs

        kspace_recon = fft2(cmg_recon)
        img_recon = complex_abs(cmg_recon)

        recons = {'kspace_recons': kspace_recon, 'cmg_recons': cmg_recon, 'img_recons': img_recon}

        if self.challenge == 'multicoil':
            rss_recon = center_crop(img_recon, (self.resolution, self.resolution)) * extra_params['cmg_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons  # recons are not rescaled except rss_recons.


class PostProcessCMGIMG(nn.Module):
    """
    Outputs are expected to be the real and imaginary parts of a complex image.
    An alternative would be to directly output the absolute image.
    Residual inputs of the necessary parts are always used. There is no option to change this.
    """
    def __init__(self, challenge, output_mode='cmg', resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.output_mode = output_mode
        self.resolution = resolution

    def forward(self, output, targets, extra_params):
        if output.size(0) > 1:
            raise NotImplementedError('Batch size is expected to be 1.')

        if self.output_mode == 'cmg':
            recons = self._cmg_output(output, targets, extra_params)
        elif self.output_mode == 'img':
            recons = self._img_output(output, targets, extra_params)
        else:
            raise NotImplementedError('Invalid output mode.')

        if self.challenge == 'multicoil':
            rss_recon = center_crop(recons['img_recons'], shape=(self.resolution, self.resolution))
            rss_recon *= extra_params['cmg_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons

    @staticmethod
    def _cmg_output(cmg_output, targets, extra_params):
        cmg_target = targets['cmg_targets']
        cmg_recon = nchw_to_kspace(cmg_output)  # Assumes data was cropped already.
        assert cmg_recon.shape == cmg_target.shape, 'Reconstruction and target sizes are different.'
        assert (cmg_recon.size(-3) % 2 == 0) and (cmg_recon.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'
        cmg_recon = cmg_recon + targets['cmg_inputs']  # Residual of complex input.
        kspace_recon = fft2(cmg_recon)
        img_recon = complex_abs(cmg_recon)
        recons = {'kspace_recons': kspace_recon, 'cmg_recons': cmg_recon, 'img_recons': img_recon}
        return recons

    @staticmethod
    def _img_output(img_output, targets, extra_params):
        img_target = targets['img_targets']
        img_recon = F.relu(img_output + targets['img_inputs'])  # Residual of image input. Also removes negative values.
        assert img_recon.shape == img_target.shape, 'Reconstruction and target sizes are different.'
        recons = {'img_recons': img_recon}
        return recons


class PostProcessIMG(nn.Module):
    def __init__(self, challenge, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.resolution = resolution

    def forward(self, img_output, targets, extra_params):
        if img_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        img_target = targets['img_targets']
        # For removing width dimension padding. Recall that complex number form has 2 as last dim size.
        left = (img_output.size(-1) - img_target.size(-1)) // 2
        right = left + img_target.size(-1)

        # Cropping width dimension by pad.
        img_recon = F.relu(img_output[..., left:right])  # Removing values below 0, which are impossible anyway.

        assert img_recon.shape == img_target.shape, 'Reconstruction and target sizes are different.'

        recons = {'img_recons': img_recon}

        if self.challenge == 'multicoil':
            rss_recon = center_crop(img_recon, (self.resolution, self.resolution)) * extra_params['img_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons  # recons are not rescaled except rss_recons.


class PostProcessWSemiKCC(nn.Module):  # Images are expected to be cropped already.
    def __init__(self, challenge, weighted=True, residual_acs=True):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.challenge = challenge
        self.weighted = weighted
        self.residual_acs = residual_acs

    def forward(self, semi_kspace_outputs, targets, extra_params):
        if semi_kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        semi_kspace_recons = nchw_to_kspace(semi_kspace_outputs)
        semi_kspace_targets = targets['semi_kspace_targets']
        assert semi_kspace_recons.shape == semi_kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (semi_kspace_recons.size(-3) % 2 == 0) and (semi_kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            semi_kspace_recons = semi_kspace_recons / weighting

        if self.residual_acs:  # Adding the semi-k-space of the ACS as a residual. Necessary due to complex cropping.
            semi_kspace_acs = targets['semi_kspace_acss']
            semi_kspace_recons = semi_kspace_recons + semi_kspace_acs

        kspace_recons = fft1(semi_kspace_recons, direction='height')
        cmg_recons = ifft1(semi_kspace_recons, direction='width')
        img_recons = complex_abs(cmg_recons)

        recons = {'semi_kspace_recons': semi_kspace_recons, 'kspace_recons': kspace_recons,
                  'cmg_recons': cmg_recons, 'img_recons': img_recons}

        if self.challenge == 'multicoil':
            rss_recons = root_sum_of_squares(img_recons, dim=1).squeeze()
            rss_recons *= extra_params['sk_scales']
            recons['rss_recons'] = rss_recons

        return recons  # Returning scaled reconstructions. Not rescaled. RSS images are rescaled.
