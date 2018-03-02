import h5py
import numpy as np

from pershombox._software_backends.dipha_adapter import _PersistenceDiagramFile
from collections import defaultdict
from .path_config import data_raw_path, data_generated_path


def get_meta_from_file_path(path):
    name_parts = path.stem.split('_')
    id = int(name_parts[0])
    freq = int(name_parts[-1])
    return id, freq


def read_dgms_from_file(path):
    with open(path, 'rb') as f:
        points = _PersistenceDiagramFile.load_from_binary_file(f).points

    dgm_0 = [p[1:] for p in points if p[0] == 0] + [p[1:] for p in points if p[0] == -1]
    dgm_1 = [p[1:] for p in points if p[0] == 1]

    return dgm_0, dgm_1


readme = \
"""
'data': access = <id>/<freq>/<barcode dim>
'target': 'target'[i] = <label of 'data'[i]> 
"""


def convert_folder_to_hdf5_file(sub_path, output_file_name):
    data_path = data_raw_path.joinpath(sub_path)
    output_path = data_generated_path.joinpath(output_file_name)

    gathered_files_by_id_by_freq = defaultdict(dict)
    for dgm_path in data_path.glob('*.diagram'):
        id, freq = get_meta_from_file_path(dgm_path)
        gathered_files_by_id_by_freq[id][freq] = dgm_path

    label_file_path = data_path.joinpath('labels.txt')
    labels = np.loadtxt(str(label_file_path))

    with h5py.File(output_path, 'w') as f:
        grp_data = f.create_group('data')
        ds_target = f.create_dataset('target', dtype='i8', shape=(len(gathered_files_by_id_by_freq),))

        f.attrs['readme'] = readme

        for id, files_by_freq in gathered_files_by_id_by_freq.items():

            ds_target[id] = labels[id] - 1
            grp_data_id = grp_data.create_group(str(id))

            for freq, file_path in files_by_freq.items():
                grp_data_id_freq = grp_data_id.create_group(str(freq))

                dim_0, dim_1 = read_dgms_from_file(file_path)

                grp_data_id_freq.create_dataset('0', data=np.array(dim_0))
                grp_data_id_freq.create_dataset('1', data=np.array(dim_1))
