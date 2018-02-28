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
            return hdf5_group.value
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

        file_path = Path(data_root_folder_path).joinpath(self.file_name)
        self._h5py_file = h5py.File(file_path, 'r')

        self._grp_data = self._h5py_file[self.data_hdf5_key]
        self._ds_target = self._h5py_file[self.target_hdf5_key]

        self._length = len(self._grp_data.keys())

    def _get_data_i(self, index: int):
        return self._grp_data[str(index)]

    def _get_target_i(self, index: int):
        return self._ds_target[index]

    def __len__(self):
        return self._length

    @property
    def targets(self):
        return self._ds_target.value

    def __del__(self):
        self._h5py_file.close()
