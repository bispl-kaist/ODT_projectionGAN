import torch
from torch import nn, optim

from pathlib import Path


class ADCGANCheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, modelG, modelD, ADC, optimizerG, optimizerD,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelD, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerD, optim.Optimizer), 'Not a Pytorch Optimizer'
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

        self.modelG = modelG
        self.modelD = modelD
        self.ADC = ADC
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
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
        save_dictG = {'model_state_dict': self.modelG.state_dict()}
        save_dictD = {'model_state_dict': self.modelD.state_dict()}
        save_dictADC = {'model_state_dict': self.ADC.state_dict()}
        save_dictG.update(save_kwargs)
        save_dictD.update(save_kwargs)
        save_dictADC.update(save_kwargs)
        save_pathG = self.ckpt_path / (f'ckpt_G{self.save_counter:03d}.tar')
        save_pathD = self.ckpt_path / (f'ckpt_D{self.save_counter:03d}.tar')
        save_pathADC = self.ckpt_path / (f'ckpt_ADC{self.save_counter:03d}.tar')

        torch.save(save_dictG, save_pathG)
        torch.save(save_dictD, save_pathD)
        torch.save(save_dictADC, save_pathADC)
        print(f'Saved Checkpoint to {save_pathG}{save_pathD}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG} {save_pathD}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG} {save_pathD}', file=file)

        self.record_dict[self.save_counter] = save_pathG

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG, save_pathD, save_pathADC

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG, save_pathD, save_pathADC = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dirG, load_dirD, load_dirADC, load_optimizer=True):
        save_dictG = torch.load(load_dirG)
        save_dictD = torch.load(load_dirD)
        save_dictADC = torch.load(load_dirADC)

        self.modelG.load_state_dict(save_dictG['model_state_dict'])
        self.modelD.load_state_dict(save_dictD['model_state_dict'])
        self.modelADC.load_state_dict(save_dictADC['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG}')

        if load_optimizer:
            self.optimizerG.load_state_dict(save_dictG['optimizer_state_dict'])
            self.optimizerD.load_state_dict(save_dictD['optimizer_state_dict'])
            self.optimizerADC.load_state_dict(save_dictADC['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG}')

    def load_G(self, load_dirG, load_optimizer=True):
        save_dictG = torch.load(load_dirG)

        self.modelG.load_state_dict(save_dictG['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG}')

        if load_optimizer:
            self.optimizerG.load_state_dict(save_dictG['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


class GANCheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, modelG, modelD, optimizerG, optimizerD,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelD, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerD, optim.Optimizer), 'Not a Pytorch Optimizer'
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

        self.modelG = modelG
        self.modelD = modelD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
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
        save_dictG = {'model_state_dict': self.modelG.state_dict()}
        save_dictD = {'model_state_dict': self.modelD.state_dict()}
        save_dictG.update(save_kwargs)
        save_dictD.update(save_kwargs)
        save_pathG = self.ckpt_path / (f'ckpt_G{self.save_counter:03d}.tar')
        save_pathD = self.ckpt_path / (f'ckpt_D{self.save_counter:03d}.tar')

        torch.save(save_dictG, save_pathG)
        torch.save(save_dictD, save_pathD)
        print(f'Saved Checkpoint to {save_pathG}{save_pathD}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG} {save_pathD}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG} {save_pathD}', file=file)

        self.record_dict[self.save_counter] = save_pathG

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG, save_pathD

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG, save_pathD = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dirG, load_dirD, load_optimizer=True):
        save_dictG = torch.load(load_dirG)
        save_dictD = torch.load(load_dirD)

        self.modelG.load_state_dict(save_dictG['model_state_dict'])
        self.modelD.load_state_dict(save_dictD['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG}')

        if load_optimizer:
            self.optimizerG.load_state_dict(save_dictG['optimizer_state_dict'])
            self.optimizerD.load_state_dict(save_dictD['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG}')

    def load_G(self, load_dirG, load_optimizer=True):
        save_dictG = torch.load(load_dirG)

        self.modelG.load_state_dict(save_dictG['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG}')

        if load_optimizer:
            self.optimizerG.load_state_dict(save_dictG['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


class UnrollCheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, modelG1, modelG2, modelG3, modelD, optimizerG, optimizerD,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelD, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerD, optim.Optimizer), 'Not a Pytorch Optimizer'
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

        self.modelG1 = modelG1
        self.modelG2 = modelG2
        self.modelG3 = modelG3
        self.modelD = modelD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
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
        save_dictG1 = {'model_state_dict': self.modelG1.state_dict()}
        save_dictG2 = {'model_state_dict': self.modelG2.state_dict()}
        save_dictG3 = {'model_state_dict': self.modelG3.state_dict()}
        save_dictD = {'model_state_dict': self.modelD.state_dict()}
        save_dictG1.update(save_kwargs)
        save_dictG2.update(save_kwargs)
        save_dictG3.update(save_kwargs)
        save_dictD.update(save_kwargs)
        save_pathG1 = self.ckpt_path / (f'ckpt_G1{self.save_counter:03d}.tar')
        save_pathG2 = self.ckpt_path / (f'ckpt_G2{self.save_counter:03d}.tar')
        save_pathG3 = self.ckpt_path / (f'ckpt_G3{self.save_counter:03d}.tar')
        save_pathD = self.ckpt_path / (f'ckpt_D{self.save_counter:03d}.tar')

        torch.save(save_dictG1, save_pathG1)
        torch.save(save_dictG2, save_pathG2)
        torch.save(save_dictG3, save_pathG3)
        torch.save(save_dictD, save_pathD)
        print(f'Saved Checkpoint to {save_pathG1}{save_pathD}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG1} {save_pathD}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG1} {save_pathD}', file=file)

        self.record_dict[self.save_counter] = save_pathG1

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG1, save_pathG2, save_pathG3, save_pathD

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG1, save_pathG2, save_pathG3, save_pathD = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG1, save_pathG2, save_pathG3, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dirG1, load_dirG2, load_dirG3, load_dirD, load_optimizer=True):
        save_dictG1 = torch.load(load_dirG1)
        save_dictG2 = torch.load(load_dirG2)
        save_dictG3 = torch.load(load_dirG3)
        save_dictD = torch.load(load_dirD)

        self.modelG1.load_state_dict(save_dictG1['model_state_dict'])
        self.modelG2.load_state_dict(save_dictG2['model_state_dict'])
        self.modelG3.load_state_dict(save_dictG3['model_state_dict'])
        self.modelD.load_state_dict(save_dictD['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG1}')

        if load_optimizer:
            self.optimizerG.load_state_dict(save_dictG1['optimizer_state_dict'])
            self.optimizerD.load_state_dict(save_dictD['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG1}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


class cycleGANCheckpointManager:
    def __init__(self, modelG_uf, modelG_fu, modelD, optimizerG, optimizerD,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelD, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerD, optim.Optimizer), 'Not a Pytorch Optimizer'
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

        self.modelG_uf = modelG_uf
        self.modelG_fu = modelG_fu
        self.modelD = modelD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
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
        save_dictG_uf = {'model_state_dict': self.modelG_uf.state_dict()}
        save_dictG_fu = {'model_state_dict': self.modelG_fu.state_dict()}
        save_dictD = {'model_state_dict': self.modelD.state_dict()}
        save_dictG_uf.update(save_kwargs)
        save_dictG_fu.update(save_kwargs)
        save_dictD.update(save_kwargs)
        save_pathG_uf = self.ckpt_path / (f'ckpt_G_uf{self.save_counter:03d}.tar')
        save_pathG_fu = self.ckpt_path / (f'ckpt_G_fu{self.save_counter:03d}.tar')
        save_pathD = self.ckpt_path / (f'ckpt_D{self.save_counter:03d}.tar')

        torch.save(save_dictG_uf, save_pathG_uf)
        torch.save(save_dictG_fu, save_pathG_fu)
        torch.save(save_dictD, save_pathD)

        print(f'Saved Checkpoint to {str(self.ckpt_path)}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG_uf} {save_pathG_fu} {save_pathD}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG_uf} {save_pathG_fu} {save_pathD}', file=file)

        self.record_dict[self.save_counter] = save_pathG_uf

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG_uf, save_pathG_fu, save_pathD

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        # save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG_uf, save_pathG_fu, save_pathD = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG_uf, save_pathG_fu, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dirG_uf, load_dirG_fu, load_dirD, load_optimizer=True):
        save_dictG_uf = torch.load(load_dirG_uf)
        save_dictG_fu = torch.load(load_dirG_fu)
        save_dictD = torch.load(load_dirD)

        self.modelG_uf.load_state_dict(save_dictG_uf['model_state_dict'])
        self.modelG_fu.load_state_dict(save_dictG_fu['model_state_dict'])
        self.modelD.load_state_dict(save_dictD['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG_uf}')

        if load_optimizer:
            self.optimizerG_uf.load_state_dict(save_dictG_uf['optimizer_state_dict'])
            self.optimizerG_fu.load_state_dict(save_dictG_fu['optimizer_state_dict'])
            self.optimizerD.load_state_dict(save_dictD['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG_uf}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


class cycleGANCheckpointManager_comparison:
    def __init__(self, modelG_DtoF, modelG_FtoD, modelD_Full, modelD_Down, optimizerG_DtoF,
                 optimizerG_FtoD, optimizerD_Full, optimizerD_Down,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelG_DtoF, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerG_DtoF, optim.Optimizer), 'Not a Pytorch Optimizer'
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

        self.modelG_DtoF = modelG_DtoF
        self.modelG_FtoD = modelG_FtoD
        self.modelD_Full = modelD_Full
        self.modelD_Down = modelD_Down

        self.optimizerG_DtoF = optimizerG_DtoF
        self.optimizerG_FtoD = optimizerG_FtoD
        self.optimizerD_Full = optimizerD_Full
        self.optimizerD_Down = optimizerD_Down

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
        save_dictG_DtoF = {'model_state_dict': self.modelG_DtoF.state_dict()}
        save_dictG_FtoD = {'model_state_dict': self.modelG_FtoD.state_dict()}
        save_dictD_Full = {'model_state_dict': self.modelD_Full.state_dict()}
        save_dictD_Down = {'model_state_dict': self.modelD_Down.state_dict()}

        save_dictG_DtoF.update(save_kwargs)
        save_dictG_FtoD.update(save_kwargs)
        save_dictD_Full.update(save_kwargs)
        save_dictD_Down.update(save_kwargs)

        save_pathG_DtoF = self.ckpt_path / (f'ckpt_G_DtoF{self.save_counter:03d}.tar')
        save_pathG_FtoD = self.ckpt_path / (f'ckpt_G_FtoD{self.save_counter:03d}.tar')
        save_pathD_Full = self.ckpt_path / (f'ckpt_D_Full{self.save_counter:03d}.tar')
        save_pathD_Down = self.ckpt_path / (f'ckpt_D_Down{self.save_counter:03d}.tar')

        torch.save(save_dictG_DtoF, save_pathG_DtoF)
        torch.save(save_dictG_FtoD, save_pathG_FtoD)
        torch.save(save_dictD_Full, save_pathD_Full)
        torch.save(save_dictD_Down, save_pathD_Down)

        print(f'Saved Checkpoint to {str(self.ckpt_path)}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG_DtoF}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG_DtoF}', file=file)

        self.record_dict[self.save_counter] = save_pathG_DtoF

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG_DtoF, save_pathG_FtoD, save_pathD_Full, save_pathD_Down

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        # save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG_DtoF, save_pathG_FtoD, save_pathD_Full, \
            save_pathD_Down = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG_DtoF, save_pathG_FtoD, save_pathD_Full, save_pathD_Down, is_best

    def load(self, load_dirG_DtoF, load_dirG_FtoD, load_dirD_Full, load_dirD_Down, load_optimizer=True):
        save_dictG_DtoF = torch.load(load_dirG_DtoF)
        save_dictG_FtoD = torch.load(load_dirG_FtoD)
        save_dictD_Full = torch.load(load_dirD_Full)
        save_dictD_Down = torch.load(load_dirD_Down)

        self.modelG_DtoF.load_state_dict(save_dictG_DtoF['model_state_dict'])
        self.modelG_FtoD.load_state_dict(save_dictG_FtoD['model_state_dict'])
        self.modelD_Full.load_state_dict(save_dictD_Full['model_state_dict'])
        self.modelD_Down.load_state_dict(save_dictD_Down['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG_DtoF}')

        if load_optimizer:
            self.optimizerG_DtoF.load_state_dict(save_dictG_DtoF['optimizer_state_dict'])
            self.optimizerG_FtoD.load_state_dict(save_dictG_FtoD['optimizer_state_dict'])
            self.optimizerD_Full.load_state_dict(save_dictD_Full['optimizer_state_dict'])
            self.optimizerD_Down.load_state_dict(save_dictD_Down['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG_DtoF}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


class cycleGANprojCheckpointManager:
    def __init__(self, modelG_DtoF, modelG_FtoD, modelD_Full, modelD_Down, modelD_proj, optimizerG_DtoF,
                 optimizerG_FtoD, optimizerD_Full, optimizerD_Down, optimizerD_proj,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelG_DtoF, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerG_DtoF, optim.Optimizer), 'Not a Pytorch Optimizer'
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

        self.modelG_DtoF = modelG_DtoF
        self.modelG_FtoD = modelG_FtoD
        self.modelD_Full = modelD_Full
        self.modelD_Down = modelD_Down
        self.modelD_proj = modelD_proj

        self.optimizerG_DtoF = optimizerG_DtoF
        self.optimizerG_FtoD = optimizerG_FtoD
        self.optimizerD_Full = optimizerD_Full
        self.optimizerD_Down = optimizerD_Down
        self.optimizerD_proj = optimizerD_proj

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
        save_dictG_DtoF = {'model_state_dict': self.modelG_DtoF.state_dict()}
        save_dictG_FtoD = {'model_state_dict': self.modelG_FtoD.state_dict()}
        save_dictD_Full = {'model_state_dict': self.modelD_Full.state_dict()}
        save_dictD_Down = {'model_state_dict': self.modelD_Down.state_dict()}
        save_dictD_proj = {'model_state_dict': self.modelD_proj.state_dict()}

        save_dictG_DtoF.update(save_kwargs)
        save_dictG_FtoD.update(save_kwargs)
        save_dictD_Full.update(save_kwargs)
        save_dictD_Down.update(save_kwargs)
        save_dictD_proj.update(save_kwargs)

        save_pathG_DtoF = self.ckpt_path / (f'ckpt_G_DtoF{self.save_counter:03d}.tar')
        save_pathG_FtoD = self.ckpt_path / (f'ckpt_G_FtoD{self.save_counter:03d}.tar')
        save_pathD_Full = self.ckpt_path / (f'ckpt_D_Full{self.save_counter:03d}.tar')
        save_pathD_Down = self.ckpt_path / (f'ckpt_D_Down{self.save_counter:03d}.tar')
        save_pathD_proj = self.ckpt_path / (f'ckpt_D_proj{self.save_counter:03d}.tar')

        torch.save(save_dictG_DtoF, save_pathG_DtoF)
        torch.save(save_dictG_FtoD, save_pathG_FtoD)
        torch.save(save_dictD_Full, save_pathD_Full)
        torch.save(save_dictD_Down, save_pathD_Down)
        torch.save(save_dictD_proj, save_pathD_proj)

        print(f'Saved Checkpoint to {str(self.ckpt_path)}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG_DtoF}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG_DtoF}', file=file)

        self.record_dict[self.save_counter] = save_pathG_DtoF

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG_DtoF, save_pathG_FtoD, save_pathD_Full, save_pathD_Down, save_pathD_proj

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        # save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG_DtoF, save_pathG_FtoD, save_pathD_Full, \
            save_pathD_Down, save_pathD_proj = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG_DtoF, save_pathG_FtoD, save_pathD_Full, save_pathD_Down, save_pathD_proj, is_best

    def load(self, load_dirG_DtoF, load_dirG_FtoD, load_dirD_Full, load_dirD_Down, load_dirD_proj, load_optimizer=True):
        save_dictG_DtoF = torch.load(load_dirG_DtoF)
        save_dictG_FtoD = torch.load(load_dirG_FtoD)
        save_dictD_Full = torch.load(load_dirD_Full)
        save_dictD_Down = torch.load(load_dirD_Down)
        save_dictD_proj = torch.load(load_dirD_proj)

        self.modelG_DtoF.load_state_dict(save_dictG_DtoF['model_state_dict'])
        self.modelG_FtoD.load_state_dict(save_dictG_FtoD['model_state_dict'])
        self.modelD_Full.load_state_dict(save_dictD_Full['model_state_dict'])
        self.modelD_Down.load_state_dict(save_dictD_Down['model_state_dict'])
        self.modelD_proj.load_state_dict(save_dictD_proj['model_state_dict'])
        print(f'Loaded model parameters from {load_dirG_DtoF}')

        if load_optimizer:
            self.optimizerG_DtoF.load_state_dict(save_dictG_DtoF['optimizer_state_dict'])
            self.optimizerG_FtoD.load_state_dict(save_dictG_FtoD['optimizer_state_dict'])
            self.optimizerD_Full.load_state_dict(save_dictD_Full['optimizer_state_dict'])
            self.optimizerD_Down.load_state_dict(save_dictD_Down['optimizer_state_dict'])
            self.optimizerD_proj.load_state_dict(save_dictD_proj['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dirG_DtoF}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


def load_gan_model_from_checkpoint(model, load_dir, strict=False):
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