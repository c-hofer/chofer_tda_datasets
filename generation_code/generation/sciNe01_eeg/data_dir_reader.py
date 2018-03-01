import numpy as np
import h5py
import glob
import os
from collections import namedtuple


GROUP_IDS = ['control', 'patient']
STR_TO_INT_GROUP_DICT = {str_label: i for i, str_label in enumerate(GROUP_IDS)}


LABEL_IDS = ['IFOT', 'IHND', 'MFOT', 'MHND', 'OFOT', 'OHND', 'REST']
STR_TO_INT_LABELS_DICT = {str_label: i for i, str_label in enumerate(LABEL_IDS)}
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


def int_group_from_str_group(group: str):
    assert isinstance(group, str)
    return STR_TO_INT_GROUP_DICT[group]


def str_group_from_int_group(group: int):
    assert isinstance(group, int)
    return GROUP_IDS[group]


def int_label_from_str_label(label: str):
    assert isinstance(label, str)
    return STR_TO_INT_LABELS_DICT[label]


def str_label_from_int_label(label: int):
    assert isinstance(label, int)
    return LABEL_IDS[label]


def down_sample_from_1000_to_250_timestamps(data):
    assert data.shape == (1000, 256)
    data_slices = [data[i:i + 4, :] for i in range(0, 1000, 4)]
    aggregated_slices = [x.mean(axis=0) for x in data_slices]
    return np.stack(aggregated_slices, axis=0)


class SciNe01DataDirReader:
    group_ids = GROUP_IDS
    label_ids = LABEL_IDS

    sample_def = namedtuple('sample_def', ('file_path', 'subject_id', 'label', 'group', 'run', 'sub_run'))

    def __init__(self, data_dir: str,
                 omit_sub_run_0=True,
                 down_sample_higher_resolution_samples=True):
        self.down_sample_higher_resolution_samples = down_sample_higher_resolution_samples
        self.data_dir = str(data_dir)
        assert os.path.isdir(self.data_dir)
        self.omit_sub_run_0 = omit_sub_run_0

        self._sample_defs = self._init_list_of_sample_defs()

    def _init_list_of_sample_defs(self):
        list_of_sample_defs = []

        file_paths_metas = glob.glob(os.path.join(self.data_dir, '*.mat'))
        file_paths_metas = [(os.path.normpath(p),
                       self._meta_info_from_file_path(p)) for p in file_paths_metas]
        sorted(file_paths_metas, key=lambda x: x[1]['subject_id'])

        for path, meta in file_paths_metas:

            subject_id = meta['subject_id']
            group = meta['group']

            # one subject has many runs with different labels ...
            for label in self.label_ids:

                # ... one subject has per label 25 runs ...
                for run in range(25):

                    # ... divided into 6 sub-runs
                    for sub_run in range(6):
                        if self.omit_sub_run_0 and sub_run == 0:
                            continue

                        list_of_sample_defs.append(self.sample_def(path, subject_id, label, group, run, sub_run))

        return list_of_sample_defs

    def _meta_info_from_file_path(self, file_name: str):
        name = os.path.basename(file_name)
        meta = {}

        for g_id in self.group_ids:
            if g_id in name:
                meta['group'] = g_id
                meta['subject_id'] = name.split('.mat')[0]

        return meta

    @property
    def labels(self):
        labels = [self._sample_defs[i].label for i in range(len(self))]
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

        x = data[s, :]

        meta = {
            'subject_id': sample_def.subject_id,
            'group': sample_def.group,
            'label': sample_def.label,
            'run': sample_def.run,
            'sub_run': sample_def.sub_run
            }

        assert x.shape[0] == 250 or x.shape[0] == 1000
        if self.down_sample_higher_resolution_samples and x.shape[0] == 1000:
            x = down_sample_from_1000_to_250_timestamps(x)

        return x, meta
