import h5py
from pathlib import Path


class SupervisedDataset(object):
    def __init__(self,
                 data_transforms: [] = None,
                 target_transforms: [] = None):

        self.data_transforms = list(data_transforms) if data_transforms is not None else []
        self.target_transforms = list(target_transforms) if target_transforms is not None else []

    def _get_data_i(self, index: int):
        raise NotImplementedError()

    def _get_target_i(self, index: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        index = int(index)
        x = self._get_data_i(index)
        y = self._get_target_i(index)

        for t in self.data_transforms:
            x = t(x)

        for t in self.target_transforms:
            y = t(y)

        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def targets(self):
        raise NotImplementedError()


def hdf5_group_to_dict(hdf5_group):
        if isinstance(hdf5_group, h5py.Dataset):
            return hdf5_group[()]
        else:
            return {k: hdf5_group_to_dict(v) for k, v in hdf5_group.items()}


class Hdf5SupervisedDatasetOneFile(SupervisedDataset):
    file_name = None
    google_drive_id = None

    data_hdf5_key = 'data'
    target_hdf5_key = 'target'

    def __init__(self,
                 data_root_folder_path: str,
                 data_transforms: [] = None,
                 target_transforms: [] = None
                 ):
        super().__init__(data_transforms=data_transforms,
                         target_transforms=target_transforms)

        self.file_path = Path(data_root_folder_path).joinpath(self.file_name)

    @property
    def _h5py_file(self):
        return h5py.File(self.file_path, 'r')

    @property
    def _grp_data(self):
        return self._h5py_file[self.data_hdf5_key]

    @property
    def _ds_target(self):
        return self._h5py_file[self.target_hdf5_key]

    def _get_data_i(self, index: int):
        return self._h5py_file[self.data_hdf5_key][str(index)]

    def _get_target_i(self, index: int):
        return self._h5py_file[self.target_hdf5_key][index]

    def __len__(self):
        return len(self._h5py_file[self.data_hdf5_key].keys())

    @property
    def targets(self):
        return self._h5py_file[self.target_hdf5_key][()]

    @property
    def readme(self):
        if 'readme' in self._h5py_file.attrs:
            return self._h5py_file.attrs['readme']
