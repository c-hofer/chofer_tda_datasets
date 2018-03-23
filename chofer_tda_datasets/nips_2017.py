"""
This module contains a nice wrapper for the persistence barcode data sets used in

@inproceedings{Hofer17c,
  author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
  title     = {Deep Learning with Topological Signatures},
  booktitle = {NIPS},
  year      = 2017}

The code how the barcodes were generated from the origional data sets can be found on

    https://github.com/c-hofer/nips2017.

"""
import h5py
import numpy as np
import os
import os.path as pth

from .utils.download import download_file_from_google_drive


# region Provider


class ProviderError(Exception):
    pass


class NameSpace:
    pass


class Provider:
    _serial_str_keys = NameSpace()
    _serial_str_keys.data_views = 'data_views'
    _serial_str_keys.str_2_int_label_map = 'str_2_int_label_map'
    _serial_str_keys.meta_data = 'meta_data'

    def __init__(self, data_views=None, str_2_int_label_map=None, meta_data=None):
        data_views = {} if data_views is None else data_views
        meta_data = {} if meta_data is None else data_views
        self.data_views = data_views
        self.str_2_int_label_map = str_2_int_label_map
        self.meta_data = meta_data
        self._cache = NameSpace()

    def add_view(self, name_of_view, view):
        assert type(name_of_view) is str
        assert isinstance(view, dict)
        assert all([type(label) is str for label in view.keys()])
        assert name_of_view not in self.data_views

        self.data_views[name_of_view] = view

    def add_str_2_int_label_map(self, label_map):
        assert isinstance(label_map, dict)
        assert all([type(str_label) is str for str_label in label_map.keys()])
        assert all([type(int_label) is int for int_label in label_map.values()])
        self.str_2_int_label_map = label_map

    def add_meta_data(self, meta_data):
        assert isinstance(meta_data, dict)
        self.meta_data = meta_data

    def _check_views_are_consistent(self):
        if len(self.data_views) > 0:
            first_view = next(iter(self.data_views.values()))

            # Check if every view has the same number of labels.
            lenghts_same = [len(first_view) == len(view) for view in self.data_views.values()]
            if not all(lenghts_same):
                raise ProviderError('Not all views have same amount of label groups.')

            # Check if every view has the same labels.
            labels_same = [set(first_view.keys()) == set(view.keys()) for view in self.data_views.values()]
            if not all(labels_same):
                raise ProviderError('Not all views have the same labels in their label groups.')

            # Check if every label group has the same number of subjects in each view.
            labels = first_view.keys()
            for k in labels:
                label_groups_cons = [set(first_view[k].keys()) == set(view[k].keys()) for view in
                                     self.data_views.values()]
                if not all(label_groups_cons):
                    raise ProviderError('There is some inconsistency in the labelgroups.' \
                                        + ' Not the same subject ids in each view for label {}'.format(k))

    def _check_str_2_int_labelmap(self):
        """
        assumption: _check_views_are_consistent allready called.
        """
        first_view = list(self.data_views.values())[0]

        # Check if the labels are the same.
        if not set(self.str_2_int_label_map.keys()) == set(first_view.keys()):
            raise ProviderError('self.str_2_int_label_map has not the same labels as the data views.')

        # Check if int labels are int.
        if not all([type(v) is int for v in self.str_2_int_label_map.values()]):
            raise ProviderError('Labels in self.str_2_int_label have to be of type int.')

    def _check_state_for_serialization(self):
        if len(self.data_views) == 0:
            raise ProviderError('Provider must have at least one view.')

        self._check_views_are_consistent()

        if self.str_2_int_label_map is not None:
            self._check_str_2_int_labelmap()

    def _prepare_state_for_serialization(self):
        self._check_state_for_serialization()

        if self.str_2_int_label_map is None:
            self.str_2_int_label_map = {}
            first_view = list(self.data_views.values())[0]

            for i, label in enumerate(first_view):
                self.str_2_int_label_map[label] = i + 1

    def dump_as_h5(self, file_path):
        self._prepare_state_for_serialization()

        with h5py.File(file_path, 'w') as file:
            data_views_grp = file.create_group(self._serial_str_keys.data_views)

            for view_name, view in self.data_views.items():
                view_grp = data_views_grp.create_group(view_name)

                for label, label_subjects in view.items():
                    label_grp = view_grp.create_group(label)

                    for subject_id, subject_values in label_subjects.items():
                        label_grp.create_dataset(subject_id, data=subject_values)

            label_map_grp = file.create_group(self._serial_str_keys.str_2_int_label_map)
            for k, v in self.str_2_int_label_map.items():
                # since the lua hdf5 implementation seems to have issues reading scalar values we
                # dump the label as 1 dimensional tuple.
                label_map_grp.create_dataset(k, data=(v,))

            meta_data_group = file.create_group(self._serial_str_keys.meta_data)
            for k, v in self.meta_data.items():
                if type(v) is str:
                    v = np.string_(v)
                    dset = meta_data_group.create_dataset(k, data=v)
                else:
                    meta_data_group.create_dataset(k, data=v)

    def read_from_h5(self, file_path):
        with h5py.File(file_path, 'r') as file:
            # load data_views
            data_views = dict(file[self._serial_str_keys.data_views])
            for view_name, view in data_views.items():
                view = dict(view)
                data_views[view_name] = view

                for label, label_group in view.items():
                    label_group = dict(label_group)
                    view[label] = label_group

                    for subject_id, value in label_group.items():
                        label_group[subject_id] = file[self._serial_str_keys.data_views][view_name][label][subject_id][
                            ()]

            self.data_views = data_views

            # load str_2_int_label_map
            str_2_int_label_map = dict(file[self._serial_str_keys.str_2_int_label_map])
            for str_label, str_to_int in str_2_int_label_map.items():
                str_2_int_label_map[str_label] = str_to_int[()]

            self.str_2_int_label_map = str_2_int_label_map
            for k, v in self.str_2_int_label_map.items():
                self.str_2_int_label_map[k] = int(v[0])

            # load meta_data
            meta_data = dict(file[self._serial_str_keys.meta_data])
            for k, v in meta_data.items():
                meta_data[k] = v[()]
            self.meta_data = meta_data

        return self

    def select_views(self, views: [str]):
        data_views = {}
        for view in views:
            data_views[view] = self.data_views[view]

        return Provider(data_views=data_views, str_2_int_label_map=self.str_2_int_label_map, meta_data=self.meta_data)

    @property
    def sample_id_to_label_map(self):
        if not hasattr(self._cache, 'sample_id_to_label_map'):
            self._cache.sample_id_to_label_map = {}
            for label, label_data in self.data_views[self.view_names[0]].items():
                for sample_id in label_data:
                    self._cache.sample_id_to_label_map[sample_id] = label

        return self._cache.sample_id_to_label_map

    @property
    def view_names(self):
        return list(self.data_views.keys())

    @property
    def labels(self):
        return list(self.data_views[self.view_names[0]].keys())

    @property
    def sample_labels(self):
        for i in range(len(self)):
            _, label = self[i]
            yield label

    @property
    def sample_ids(self):
        if not hasattr(self._cache, 'sample_ids'):
            first_view = self.data_views[self.view_names[0]]
            self._cache.sample_ids = sum([list(label_group.keys()) for label_group in first_view.values()], [])

        return self._cache.sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample_label = self.sample_id_to_label_map[sample_id]

        x = {}

        for view_name, view_data in self.data_views.items():
            x[view_name] = view_data[sample_label][sample_id]

        return x, sample_label


# endregion


# region datasets


_data_set_name_2_provider_google_drive_id = \
    {
        'animal': '0BxHF82gaPzgSSWIxNmJBRFJzcmM',
        'mpeg7': '0BxHF82gaPzgSU3lPWDNEVHhNR3M',
        'reddit_5K': '0BxHF82gaPzgSZDdFWDU3S29hdm8',
        'reddit_12K': '0BxHF82gaPzgSd0d4WDNYVnN4dEU',
    }


_data_set_name_2_provider_name = \
    {
        'animal': 'npht_animal_32dirs.h5',
        'mpeg7': 'npht_mpeg7_32dirs.h5',
        'reddit_5K': 'reddit_5K.h5',
        'reddit_12K': 'reddit_12K.h5'
    }


class DataSetException(Exception):
    pass


class DataSetBase:
    google_drive_provider_id = None
    provider_file_name = None

    def __init__(self, root_dir: str, download=True, sample_transforms: list=None):
        sample_transforms = [] if sample_transforms is None else sample_transforms
        self.root_dir = pth.normpath(root_dir)
        self.data_transforms = sample_transforms
        self._provider = None
        self.integer_labels = True

        provider_exists = pth.isfile(self._provider_file_path)

        if not provider_exists:
            print("Did not find data in {}!".format(self.root_dir))

            if download:
                print("Downloading ... ")
                if not pth.isdir(self.root_dir):
                    os.mkdir(self.root_dir)

                download_file_from_google_drive(self.google_drive_provider_id,
                                                self._provider_file_path)

        provider_exists = pth.isfile(self._provider_file_path)
        if provider_exists:
            print('Found data!')
            self._provider = Provider()
            self._provider.read_from_h5(self._provider_file_path)

        else:
            raise DataSetException("Cannot find data in {}.".format(self.root_dir))

        self.str_2_int_label = {str_label: int_label for int_label, str_label in enumerate(self._provider.labels)}

    @property
    def _provider_file_path(self):
        return pth.join(self.root_dir, self.provider_file_name)

    def __getitem__(self, item):
        x, y = self._provider[item]

        for t in self.data_transforms:
            x = t(x)

        if self.integer_labels:
            y = self.str_2_int_label[y]

        return x, y

    def __len__(self):
        return len(self._provider)

    @property
    def labels(self):
        return [self[i][1] for i in range(len(self))]


class Animal(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSSWIxNmJBRFJzcmM'
    provider_file_name = 'npht_animal_32dirs.h5'


class Mpeg7(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSU3lPWDNEVHhNR3M'
    provider_file_name = 'npht_mpeg7_32dirs.h5'


class Reddit_5K(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSZDdFWDU3S29hdm8'
    provider_file_name = 'reddit_5K.h5'


class Reddit_12K(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSd0d4WDNYVnN4dEU'
    provider_file_name = 'reddit_12K.h5'


# endregion