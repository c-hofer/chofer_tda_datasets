import numpy as np
import h5py
import glob
import os
from collections import namedtuple

GROUP_IDS = ['control', 'patient']
LABEL_IDS = ['IFOT', 'IHND', 'MFOT', 'MHND', 'OFOT', 'OHND', 'REST']
STR_TO_INT_LABLES_DICT = {str_label: i for i, str_label in enumerate(LABEL_IDS)}
SENSOR_CONFIGURATIONS = \
    {
        # x-1 as original sensor counting started with 1 (not with 0)
        'all': [x - 1 for x in range(1, 257)],
        'low_resolution_sensorimotor_cortex': [x - 1 for x in [59, 183]],
        'low_resolution_whole_head': [x - 1 for x in
                                      [2, 18, 21, 26, 36, 37, 47, 59, 69, 87, 96, 101, 116, 126, 150, 153, 170, 183,
                                       202, 224]],
        'high_resolution_sensorimotor_cortex': [x - 1 for x in
                                                [59, 52, 44, 51, 43, 9, 186, 185, 184, 183, 60, 53, 45, 132, 143, 144,
                                                 155, 79, 80, 81, 131, 17, 8, 198, 197, 196, 101]],

        'high_resolution_all_but_bad_channels': [x - 1 for x in range(1, 257) if x not in
                                                 [31, 37, 32, 25, 18, 91, 102, 103, 111, 120, 133, 145, 165, 174, 187,
                                                  199, 208, 216, 112, 121, 134, 146, 156, 166, 175, 188, 200] +
                                                 list(range(225, 258))]
    }


def _int_label_from_str_label(label: str):
    return STR_TO_INT_LABLES_DICT[label]


def down_sample_from_1000_to_250_timestamps(data):
    assert data.shape == (1000, 256)
    data_slices = [data[i:i + 4, :] for i in range(0, 1000, 4)]
    aggregated_slices = [x.mean(axis=0) for x in data_slices]
    return np.stack(aggregated_slices, axis=0)


class SciNe01DataDirReader:
    group_ids = GROUP_IDS
    label_ids = LABEL_IDS

    sample_def = namedtuple('sample_def', ('file_path', 'label', 'run', 'sub_run'))

    def __init__(self, data_dir: str,
                 omit_sub_run_0=True,
                 int_labels=True,
                 down_sample_higher_resolution_samples=True):
        self.down_sample_higher_resolution_samples = down_sample_higher_resolution_samples
        self.int_labels = bool(int_labels)
        self.data_dir = str(data_dir)
        assert os.path.isdir(self.data_dir)
        self.omit_sub_run_0 = omit_sub_run_0

        self._sample_defs = self._init_list_of_sample_defs()

    def _sorted_list_file_paths(self):
        file_paths = glob.glob(os.path.join(self.data_dir, '*.mat'))
        file_paths = [os.path.normpath(p) for p in file_paths]
        sorted(file_paths, key=lambda x: self._meta_info_from_file_name(x)['id'])
        return file_paths

    def _init_list_of_sample_defs(self):
        list_of_sample_defs = []
        file_paths = self._sorted_list_file_paths()

        for p in file_paths:
            for l in self.label_ids:

                for run in range(25):
                    for sub_run in range(6):
                        if self.omit_sub_run_0 and sub_run == 0:
                            continue

                        list_of_sample_defs.append(self.sample_def(p, l, run, sub_run))

        return list_of_sample_defs

    def _meta_info_from_file_name(self, file_name: str):
        name = os.path.basename(file_name)
        meta = {}

        for g_id in self.group_ids:
            if g_id in name:
                meta['group'] = g_id
                meta['id'] = name.split('.mat')[0]

        return meta

    @property
    def labels(self):
        labels = [self._sample_defs[i].label for i in range(len(self))]

        if self.int_labels:
            labels = [_int_label_from_str_label(l) for l in labels]

        return labels

    def __len__(self):
        return len(self._sample_defs)

    def __getitem__(self, key):
        sample_def = self._sample_defs[key]

        file_path = sample_def.file_path
        f = h5py.File(file_path, 'r')

        data = f[sample_def.label].value
        data = data[:, :, sample_def.run]

        n_time_stamps = data.shape[0]
        sub_run_length = int(n_time_stamps / 6)
        s = slice(sub_run_length * sample_def.sub_run, sub_run_length * (sample_def.sub_run + 1))

        x, y = data[s, :], sample_def.label

        assert x.shape[0] == 250 or x.shape[0] == 1000
        if self.down_sample_higher_resolution_samples and x.shape[0] == 1000:
            x = down_sample_from_1000_to_250_timestamps(x)

        if self.int_labels:
            y = _int_label_from_str_label(y)

        return x, y


# TODO move to chofer_datasets package
# class BottomTopHeightFiltration:
#     def __init__(self, data_path: str, sensor_configuration='all'):
#         self.data_path = str(data_path)
#
#         self._h5_file = h5py.File(self.data_path, 'r')
#         self._grp_data = self._h5_file['data']
#         self._ds_labels = self._h5_file['labels']
#
#         self._sensors_to_take = SENSOR_CONFIGURATIONS[sensor_configuration]
#
#     def __len__(self):
#         return len(self._h5_file['labels'])
#
#     @property
#     def labels(self):
#         labels = [int(x) for x in self._h5_file['labels']]
#         return labels
# 
#     def __getitem__(self, item):
#         x = {}
#
#         grp_id = self._grp_data[str(item)]
#         for filt_name in grp_id.keys():
#             ds_dgms = grp_id[filt_name]
#             dgms = []
#
#             for sensor_i in self._sensors_to_take:
#                 x_coordinates = ds_dgms[sensor_i, 0]
#                 y_coordinates = ds_dgms[sensor_i, 1]
#
#                 dgm = np.stack([x_coordinates, y_coordinates], axis=1)
#
#                 dgms.append(dgm)
#
#             x[filt_name] = dgms
#
#         y = int(self._ds_labels[item])
#
#         return x, y
#
#     def __del__(self):
#         self._h5_file.close()
