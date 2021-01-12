import pathlib
import random
from math import ceil
from glob import glob

import h5py
from scipy.io import loadmat
from torch.utils.data import Dataset
from data.data_transforms import to_tensor
import numpy as np


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_gt=True):
        """
        Args:
            root (Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice_num' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
            use_gt (bool): Whether to load the ground truth 320x320 fully-sampled reconstructions or not.
                Very useful for reducing data I/O in k-space learning.
        """

        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.use_gt = use_gt

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = list()
        files = list(pathlib.Path(root).glob('*.h5'))

        if not files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}. This might take a minute.')

        if sample_rate < 1:
            random.shuffle(files)
            num_files = ceil(len(files) * sample_rate)
            files = files[:num_files]

        for file_name in sorted(files):
            kspace = h5py.File(file_name, mode='r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(file_name, slice_num) for slice_num in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        file_path, slice_num = self.examples[idx]
        with h5py.File(file_path, mode='r') as data:
            k_slice = data['kspace'][slice_num]
            if (self.recons_key in data) and self.use_gt:
                target_slice = data[self.recons_key][slice_num]
            else:
                target_slice = None
            return self.transform(k_slice, target_slice, data.attrs, file_path.name, slice_num)


class CustomSliceData(Dataset):

    def __init__(self, root, transform, challenge, sample_rate=1, start_slice=0, use_gt=False):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.use_gt = use_gt

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = list()
        files = list(pathlib.Path(root).iterdir())

        if not files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        if sample_rate < 1:
            random.shuffle(files)
            num_files = ceil(len(files) * sample_rate)
            files = files[:num_files]

        for file_name in sorted(files):
            kspace = h5py.File(file_name, mode='r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(file_name, slice_num) for slice_num in range(start_slice, num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        file_path, slice_num = self.examples[idx]
        with h5py.File(file_path, mode='r') as data:
            attrs = dict(data.attrs)
            k_slice = data['kspace'][slice_num]
            if (self.recons_key in data) and self.use_gt:
                target_slice = data[self.recons_key][slice_num]
            else:
                target_slice = None

        return self.transform(k_slice, target_slice, attrs, file_path.name, slice_num)


class TOFData(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*0210*/*.mat')) + list(root.glob('Input/*0325*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        # label_files = label_files + label_files.copy()  # So that the number of elements is not smaller than number of label files
        input_files = input_files + input_files.copy()
        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]
        if '0325' in input_file_path:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['label']
        else:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['input']

        if '0504' in label_file_path:
            label_slice = loadmat(label_file_path)
            label_slice = label_slice['input']
        else:
            label_slice = loadmat(label_file_path)
            label_slice = label_slice['label']

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class TOFData_double(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        label_files = label_files + label_files.copy()
        num_label_files = ceil(len(input_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)
        # self.input_files.append(str(sorted(input_files[0])))

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)
        # self.label_files.append(str(sorted(label_files[0])))

    def __len__(self):
        # Let's see if this works
        # Reducing the length of the dataset so that index error does not occur at the last index
        # Since in the __getitem__ method we are calling files[idx+1]
        return len(self.input_files) - 1

    def __getitem__(self, idx):

        input_file_path1 = self.input_files[idx]
        input_file_path2 = self.input_files[idx+1]
        label_file_path1 = self.label_files[idx]
        label_file_path2 = self.label_files[idx+1]

        if '0325' in input_file_path1:
            input_slice = loadmat(input_file_path1)
            input_slice1 = input_slice['label']
        else:
            input_slice = loadmat(input_file_path1)
            input_slice1 = input_slice['input']
        if '0325' in input_file_path2:
            input_slice = loadmat(input_file_path2)
            input_slice2 = input_slice['label']
        else:
            input_slice = loadmat(input_file_path2)
            input_slice2 = input_slice['input']

        if '0504' in label_file_path1:
            label_slice = loadmat(label_file_path1)
            label_slice1 = label_slice['input']
        else:
            label_slice = loadmat(label_file_path1)
            label_slice1 = label_slice['label']
        if '0504' in label_file_path2:
            label_slice = loadmat(label_file_path2)
            label_slice2 = label_slice['input']
        else:
            label_slice = loadmat(label_file_path2)
            label_slice2 = label_slice['label']

        return self.transform(input_slice1, input_slice2, label_slice1, label_slice2, input_file_path1, input_file_path2,
        label_file_path1, label_file_path2)


class TOFData_cs4(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*cs4*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]
        if '0325' in input_file_path:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['label']
        else:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['input']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class TOFData_cs8(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*cs8*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]
        if '0325' in input_file_path:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['label']
        else:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['input']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class TOFData_cs8_0504(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*0504/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        label_files = label_files + label_files.copy()
        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(input_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]
        if '0325' in input_file_path:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['label']
        else:
            input_slice = loadmat(input_file_path)
            input_slice = input_slice['input']

        if '0504' in label_file_path:
            label_slice = loadmat(label_file_path)
            label_slice = label_slice['input']
        else:
            label_slice = loadmat(label_file_path)
            label_slice = label_slice['label']

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class TOFData_multislice(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*0504/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        label_files = label_files + label_files.copy()
        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(input_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]

        input_slice = loadmat(input_file_path)
        input_slice = input_slice['input']

        label_slice = loadmat(label_file_path)
        label_slice = label_slice['input']

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class SAIT_cycleGAN_data(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*.mat'))
        label_files = list(root.glob('Label/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        label_files = label_files * 100
        num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]

        input_slice = loadmat(input_file_path)
        input_slice = input_slice['input']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class ODT_data(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        if str(root) == '/media/harry/mri/CT_AAPM/make_phantom/TomoGAN_db':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_ball':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_singleball':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_singleballSL':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_7':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_SheppLogan':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_multiball':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data/TomoGAN_db_multiball_34':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        elif str(root) == '/media/harry/mri1/backup/ODT/data_TomoGAN_LimitedAngle/AAPM/angle120':
            input_files = list(root.glob('Input/*.mat'))
            label_files = list(root.glob('Label/*.mat'))
        else:
            input_files = list(root.glob('Input/*/*/*.mat'))
            label_files = list(root.glob('Label/*/*/*.mat'))
        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        label_files = label_files * 4
        num_label_files = ceil(len(input_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]

        input_slice = loadmat(input_file_path)
        # input_slice = input_slice['proj']
        input_slice = input_slice['input']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']
        # label_slice = label_slice['proj']
        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class ODT_data_infer(Dataset):

    def __init__(self, root, transform, specimen_type='all', specimen_fname='all'):

        self.transform = transform
        self.root = root
        if specimen_fname == 'all':
            if specimen_type == 'all':
                input_axi_files = list(root.glob('Input/*/*/*.mat'))
            else:
                input_axi_files = list(root.glob(f'Input/{specimen_type}/*/*.mat'))
        else:
            input_axi_files = list(root.glob(f'Input/{specimen_type}/{specimen_fname}/*.mat'))

        if not input_axi_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        self.input_axi_files = list()

        for input_file_name in sorted(input_axi_files):
            input_file_name_str = str(input_file_name)
            self.input_axi_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_axi_files)

    def __getitem__(self, idx):
        input_axi_file_path = self.input_axi_files[idx]

        input_axi_slice = loadmat(input_axi_file_path)
        input_axi_slice = input_axi_slice['proj']

        pad_input_axi_slice = np.zeros([376, 376])
        pad_input_axi_slice[2:-2, 2:-2] = input_axi_slice

        return pad_input_axi_slice, input_axi_file_path


class ODT_data_phantom_infer(Dataset):

    def __init__(self, root, transform, specimen_num='all'):

        self.transform = transform
        self.root = root
        self.specimen_num = specimen_num
        self.ball_idx = [1, 3, 4, 7, 12, 14]
        if specimen_num == 'all':
            input_files = list(root.glob(f'Input/*/*.mat'))
        elif specimen_num == 'ball':
            input_files = list(root.glob(f'Input/1/*.mat'))
            for i in range(1, len(self.ball_idx)):
                input_files += list(root.glob(f'Input/{self.ball_idx[i]}/*.mat'))
        else:
            input_files = list(root.glob(f'Input/{specimen_num}/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_axi_slice = loadmat(input_file_path)
        input_axi_slice = input_axi_slice['input']

        return input_axi_slice, input_file_path


class AAPM_infer(Dataset):

    def __init__(self, root, transform, specimen_num='all'):

        self.transform = transform
        self.root = root
        self.specimen_num = specimen_num
        if specimen_num == 'all':
            input_files = list(root.glob(f'Input/*.mat'))
        else:
            input_files = list(root.glob(f'Input/ph0{self.specimen_num}*'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_axi_slice = loadmat(input_file_path)
        input_axi_slice = input_axi_slice['input']

        return input_axi_slice, input_file_path


class AAPM(Dataset):

    def __init__(self, root, transform):

        self.transform = transform
        self.root = root
        input_files = list(root.glob(f'Input/*.mat'))
        label_files = list(root.glob(f'Label/*.mat'))
        print(f'Initializing {root}.')

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            self.input_files.append(str(input_file_name))
        for label_file_name in sorted(label_files):
            self.label_files.append(str(label_file_name))

        # Comment out the three lines if 'input' and 'label' are balanced
        num_files = len(self.label_files)
        # self.label_files *= 2
        self.input_files *= 2
        self.input_files = self.input_files[:num_files]
        # self.label_files = self.label_files[:num_files]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_axi_slice = loadmat(input_file_path)
        input_axi_slice = input_axi_slice['input']

        label_file_path = self.label_files[idx]
        label_axi_slice = loadmat(label_file_path)
        label_axi_slice = label_axi_slice['label']

        return self.transform(input_axi_slice, label_axi_slice, input_file_path, label_file_path)


class ODT_data_infer_3axis(Dataset):

    def __init__(self, root, transform, specimen_type='all', specimen_fname='all'):

        self.transform = transform
        self.root = root
        if specimen_fname == 'all':
            if specimen_type == 'all':
                input_axi_files = list(root.glob('Input_infer_axi/*/*/*.mat'))
                input_cor_files = list(root.glob('Input_infer_cor/*/*/*.mat'))
                input_sag_files = list(root.glob('Input_infer_sag/*/*/*.mat'))
            else:
                input_axi_files = list(root.glob(f'Input_infer_axi/{specimen_type}/*/*.mat'))
                input_cor_files = list(root.glob(f'Input_infer_cor/{specimen_type}/*/*.mat'))
                input_sag_files = list(root.glob(f'Input_infer_sag/{specimen_type}/*/*.mat'))
        else:
            input_axi_files = list(root.glob(f'Input_infer_axi/{specimen_type}/{specimen_fname}/*.mat'))
            input_cor_files = list(root.glob(f'Input_infer_cor/{specimen_type}/{specimen_fname}/*.mat'))
            input_sag_files = list(root.glob(f'Input_infer_sag/{specimen_type}/{specimen_fname}/*.mat'))

        if not input_cor_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        self.input_axi_files = list()
        self.input_cor_files = list()
        self.input_sag_files = list()

        for input_file_name in sorted(input_axi_files):
            input_file_name_str = str(input_file_name)
            self.input_axi_files.append(input_file_name_str)
        for input_file_name in sorted(input_cor_files):
            input_file_name_str = str(input_file_name)
            self.input_cor_files.append(input_file_name_str)
        for input_file_name in sorted(input_sag_files):
            input_file_name_str = str(input_file_name)
            self.input_sag_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_cor_files)

    def __getitem__(self, idx):
        input_axi_file_path = self.input_axi_files[idx]
        input_cor_file_path = self.input_cor_files[idx]
        input_sag_file_path = self.input_sag_files[idx]

        input_axi_slice = loadmat(input_axi_file_path)
        input_axi_slice = input_axi_slice['proj']
        input_cor_slice = loadmat(input_cor_file_path)
        input_cor_slice = input_cor_slice['proj']
        input_sag_slice = loadmat(input_sag_file_path)
        input_sag_slice = input_sag_slice['proj']

        return input_axi_slice, input_cor_slice, input_sag_slice, input_axi_file_path, input_cor_file_path, input_sag_file_path


class ODT_data_infer_3axis_microbead(Dataset):

    def __init__(self, root, transform, specimen_type='all'):

        self.transform = transform
        self.root = root
        input_axi_files = list(root.glob(f'Input_microbead_axi/{specimen_type}/*.mat'))
        input_cor_files = list(root.glob(f'Input_microbead_cor/{specimen_type}/*.mat'))
        input_sag_files = list(root.glob(f'Input_microbead_sag/{specimen_type}/*.mat'))

        if not input_cor_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        self.input_axi_files = list()
        self.input_cor_files = list()
        self.input_sag_files = list()

        for input_file_name in sorted(input_axi_files):
            input_file_name_str = str(input_file_name)
            self.input_axi_files.append(input_file_name_str)
        for input_file_name in sorted(input_cor_files):
            input_file_name_str = str(input_file_name)
            self.input_cor_files.append(input_file_name_str)
        for input_file_name in sorted(input_sag_files):
            input_file_name_str = str(input_file_name)
            self.input_sag_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_cor_files)

    def __getitem__(self, idx):
        input_axi_file_path = self.input_axi_files[idx]
        input_cor_file_path = self.input_cor_files[idx]
        input_sag_file_path = self.input_sag_files[idx]

        input_axi_slice = loadmat(input_axi_file_path)
        input_axi_slice = input_axi_slice['proj']
        input_cor_slice = loadmat(input_cor_file_path)
        input_cor_slice = input_cor_slice['proj']
        input_sag_slice = loadmat(input_sag_file_path)
        input_sag_slice = input_sag_slice['proj']

        return input_axi_slice, input_cor_slice, input_sag_slice, input_axi_file_path, input_cor_file_path, input_sag_file_path


class ODT_data_infer_3axis_phantom(Dataset):

    def __init__(self, root, transform):

        self.transform = transform
        self.root = root
        input_axi_files = list(root.glob(f'Input_axi/*.mat'))
        input_cor_files = list(root.glob(f'Input_cor/*.mat'))
        input_sag_files = list(root.glob(f'Input_sag/*.mat'))

        if not input_cor_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        self.input_axi_files = list()
        self.input_cor_files = list()
        self.input_sag_files = list()

        for input_file_name in sorted(input_axi_files):
            input_file_name_str = str(input_file_name)
            self.input_axi_files.append(input_file_name_str)
        for input_file_name in sorted(input_cor_files):
            input_file_name_str = str(input_file_name)
            self.input_cor_files.append(input_file_name_str)
        for input_file_name in sorted(input_sag_files):
            input_file_name_str = str(input_file_name)
            self.input_sag_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_cor_files)

    def __getitem__(self, idx):
        input_axi_file_path = self.input_axi_files[idx]
        input_cor_file_path = self.input_cor_files[idx]
        input_sag_file_path = self.input_sag_files[idx]

        input_axi_slice = loadmat(input_axi_file_path)
        input_axi_slice = input_axi_slice['proj']
        input_cor_slice = loadmat(input_cor_file_path)
        input_cor_slice = input_cor_slice['proj']
        input_sag_slice = loadmat(input_sag_file_path)
        input_sag_slice = input_sag_slice['proj']

        return input_axi_slice, input_cor_slice, input_sag_slice, input_axi_file_path, input_cor_file_path, input_sag_file_path


class SAIT_cycleGAN_data_infer(Dataset):

    def __init__(self, root, specimen):

        self.root = root
        self.specimen = specimen
        if self.specimen == 'all':
            input_files = list(root.glob(f'Input/*.mat'))
        else:
            input_files = list(root.glob(f'Input/{specimen}/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')
        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]

        input_slice = loadmat(input_file_path)
        input_slice = input_slice['input']

        return input_slice, input_file_path


class TOFData_Axial(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        input_files = input_files + input_files.copy()  # So that the number of elements is not smaller than number of label files

        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]

        input_slice = loadmat(input_file_path)
        input_slice = input_slice['slice_im']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['slice_im']

        return input_slice, label_slice, input_file_path, label_file_path


class TOFData_Axial_Proj(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root

        input_files = list(root.glob('Input/*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))
        proj_files = list(root.glob('Label_MIP/*/*.mat'))

        if not input_files or not label_files or not proj_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        label_files = label_files + label_files.copy()
        proj_files = proj_files + proj_files.copy()

        random.shuffle(input_files)
        random.shuffle(label_files)
        random.shuffle(proj_files)
        num_label_files = ceil(len(input_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same
        proj_files = proj_files[:num_label_files]

        self.input_files = list()
        self.label_files = list()
        self.proj_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

        for proj_file_name in sorted(proj_files):
            proj_file_name_str = str(proj_file_name)
            self.proj_files.append(proj_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]
        proj_file_path = self.proj_files[idx]

        input_slice = loadmat(input_file_path)
        input_slice = input_slice['slice_im']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['slice_im']
        proj_slice = loadmat(proj_file_path)
        proj_slice = proj_slice['MIP']

        return self.transform(input_slice, label_slice, proj_slice, input_file_path, label_file_path, proj_file_path)


class TOFData_v2(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        input_files = input_files + input_files.copy()  # So that the number of elements is not smaller than number of label files

        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]

        input_slice = loadmat(input_file_path)
        input_slice = input_slice['input']
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']
        input_slice = to_tensor(input_slice)
        label_slice = to_tensor(label_slice)

        return input_slice, label_slice, input_file_path, label_file_path


class TOFData_val(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        label_files = list(root.glob('Label/*/*.mat'))

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        if sample_rate < 1:
            random.shuffle(label_files)
            num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

            label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.label_files = list()

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file_path = self.label_files[idx]
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']

        return self.transform(label_slice, label_file_path)


class TOFData_val_v2(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        label_files = list(root.glob('Label/*/*.mat'))

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')

        if sample_rate < 1:
            random.shuffle(label_files)
            num_label_files = ceil(len(label_files) * sample_rate)  # Use number of label since it is smaller

            label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        self.label_files = list()

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file_path = self.label_files[idx]
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']
        label_slice = to_tensor(label_slice)

        return label_slice, label_file_path


class TOFData_CtoA(Dataset):

    def __init__(self, root, patient, transform, sample_rate=1):

        self.transform = transform
        self.root = root  # TOF_data/train/Label
        self.patient = patient  # patient folder name e.g.) Patient2_post
        label_files = list(root.glob(f'{patient}/*.mat'))

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.label_files = list()

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file_path = self.label_files[idx]
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['label']

        label_slice = to_tensor(label_slice)

        if self.transform:
            label_slice, label_file_path = self.transform(label_slice, label_file_path)

        return label_slice, label_file_path


class TOFDataAxial(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('Input/*/*.mat'))
        label_files = list(root.glob('Label/*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        if not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing {root}.')


        random.shuffle(input_files)
        random.shuffle(label_files)
        num_label_files = ceil(len(input_files) * sample_rate)  # Use number of label since it is smaller

        input_files = input_files[:num_label_files]  # Input and label file number do not have to be the same,
        label_files = label_files[:num_label_files]  # but for the sake of simplicity, let it be same

        # self.input_files = list(x for x in input_files.iterdir() if x.is_dir())
        # self.label_files = list(x for x in label_files.iterdir() if x.is_dir())
        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        label_file_path = self.label_files[idx]

        input_slice = loadmat(input_file_path, verify_compressed_data_integrity=False)
        input_slice = input_slice['slice_im']
        label_slice = loadmat(label_file_path, verify_compressed_data_integrity=False)
        label_slice = label_slice['slice_im']
        input_slice = to_tensor(input_slice)
        label_slice = to_tensor(label_slice)

        return self.transform(input_slice, label_slice, input_file_path, label_file_path)


class TOFData_inference(Dataset):

    def __init__(self, root, patient, transform, sample_rate=1):

        self.transform = transform
        self.root = root  # TOF_data/train/Label
        self.patient = patient  # patient folder name e.g.) Patient2_post
        input_files = list(root.glob(f'{patient}/*.mat'))
        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        if '0325' in input_file_path:
            input_slice = input_slice['label']
        else:
            input_slice = input_slice['input']

        input_slice = to_tensor(input_slice)

        if self.transform:
            input_slice, input_file_path = self.transform(input_slice, input_file_path)

        return input_slice, input_file_path


class TOFData_inference_all(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root  # TOF_data/train/Label
        input_files = list(root.glob('*/*.mat'))
        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        if '0325' in input_file_path:
            input_slice = input_slice['label']
        else:
            input_slice = input_slice['input']

        input_slice = to_tensor(input_slice)

        if self.transform:
            input_slice, input_file_path = self.transform(input_slice, input_file_path)

        return input_slice, input_file_path


class TOFData_inference_all_cs4(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root
        input_files = list(root.glob('*cs4*/*.mat'))
        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        if '0325' in input_file_path:
            input_slice = input_slice['label']
        else:
            input_slice = input_slice['input']

        input_slice = to_tensor(input_slice)

        if self.transform:
            input_slice, input_file_path = self.transform(input_slice, input_file_path)

        return input_slice, input_file_path


class TOFData_inference_all_cs8(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root  # TOF_data/train/Label
        input_files = list(root.glob('*cs8_0504/*.mat'))
        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        if '0325' in input_file_path:
            input_slice = input_slice['label']
        else:
            input_slice = input_slice['input']
        # input_slice = input_slice['input']

        input_slice = to_tensor(input_slice)

        if self.transform:
            input_slice, input_file_path = self.transform(input_slice, input_file_path)

        return input_slice, input_file_path


class TOFData_inference_A(Dataset):

    def __init__(self, root, patient, transform, sample_rate=1):

        self.transform = transform
        self.root = root  # TOF_data/train/Label
        self.patient = patient  # patient folder name e.g.) Patient2_post
        input_files = list(root.glob(f'{patient}/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        input_slice = input_slice['slice_im']

        input_slice = to_tensor(input_slice)

        if self.transform:
            input_slice, input_file_path = self.transform(input_slice, input_file_path)

        return input_slice, input_file_path


class TOFData_inference_A_all(Dataset):

    def __init__(self, root, transform, sample_rate=1):

        self.transform = transform
        self.root = root  # TOF_data/train/Label
        input_files = list(root.glob('*/*.mat'))

        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        input_slice = input_slice['slice_im']

        input_slice = to_tensor(input_slice)

        if self.transform:
            input_slice, input_file_path = self.transform(input_slice, input_file_path)

        return input_slice, input_file_path