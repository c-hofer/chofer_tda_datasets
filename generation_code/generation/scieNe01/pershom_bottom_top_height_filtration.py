import multiprocessing
import os
from collections import defaultdict

import h5py
import numpy
import numpy as np
import pershombox

from .data_dir_reader import SciNe01DataDirReader
from ..path_config import data_raw_path, data_generated_path
from generation_code.generation.utils.gui import SimpleProgressCounter


def height_filtration_from_top(f_first, f_second):
    return max(-f_first, -f_second)


def heigt_filtration_from_bottom(f_first, f_second):
    return max(f_first, f_second)


def pershom_of_timeseries(timeseries, filtration):
    assert isinstance(timeseries, numpy.ndarray)
    assert timeseries.ndim == 1

    timeseries = timeseries.tolist()
    toplices = []
    filt_values = []

    for i in range(len(timeseries) - 1):
        toplices.append((i, i + 1))
        filt_value = filtration(timeseries[i], timeseries[i + 1])
        filt_values.append(filt_value)

    return pershombox.toplex_persistence_diagrams(toplices=toplices,
                                                  filtration_values=filt_values,
                                                  deessentialize=True)


def job(args):
    id, data, label = args
    dgms = defaultdict(list)
    for i_sensor in range(data.shape[1]):
        signal = data[:, i_sensor]
        signal = (signal - signal.mean()) / signal.std()

        dgms['filt_height_from_top'].append(pershom_of_timeseries(signal, height_filtration_from_top)[0])
        dgms['filt_height_from_bottom'].append(pershom_of_timeseries(signal, heigt_filtration_from_bottom)[0])

    return {'id': id,
            'dgms': dgms,
            'label': label}


def job_arg_iter(data_reader):
    assert isinstance(data_reader, SciNe01DataDirReader)
    for id in range(len(data_reader)):
        x, y = data_reader[id]
        yield id, x, y

# TODO use path_config
EGG_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scitrecs_eeg_data/')
OUTPUT_DIR = os.path.join(EGG_DATA_DIR, 'scitrecs_eeg_pershom_bottom_top_filtration.h5')


def run():
    data_reader = SciNe01DataDirReader(EGG_DATA_DIR, int_labels=True)

    progress = SimpleProgressCounter(len(data_reader))
    progress.display()
    n_cores = min(multiprocessing.cpu_count() - 1, 10)

    vlen_dtype = h5py.special_dtype(vlen=np.dtype('float32'))

    with h5py.File(OUTPUT_DIR, 'w') as h5file:

        grp_data = h5file.create_group('data')
        ds_labels = h5file.create_dataset('labels',
                                          dtype='i8',
                                          shape=(len(data_reader),))

        with multiprocessing.Pool(n_cores) as p:

            for ret_val in p.imap_unordered(job, job_arg_iter(data_reader)):
                id = ret_val['id']
                dgms = ret_val['dgms']
                label = ret_val['label']

                ds_labels[id] = label
                grp_id = grp_data.create_group(str(id))

                for filt_name, dgm_list in dgms.items():
                    ds_id_filt = grp_id.create_dataset(filt_name,
                                                       shape=(len(dgm_list), 2),
                                                       dtype=vlen_dtype)

                    for i_sensor, dgm in enumerate(dgm_list):
                        ds_id_filt[i_sensor, 0] = [x[0] for x in dgm]
                        ds_id_filt[i_sensor, 1] = [x[1] for x in dgm]

                progress.trigger_progress()


if __name__ == "__main__":
    run()