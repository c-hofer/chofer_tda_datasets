import multiprocessing

import h5py
import numpy
import numpy as np
import pershombox

from collections import defaultdict
from .data_dir_reader import SciNe01DataDirReader, \
    int_group_from_str_group, \
    int_label_from_str_label, \
    LABEL_IDS, \
    GROUP_IDS
from ..path_config import data_raw_path, data_generated_path
from ..utils.gui import SimpleProgressCounter
from .data_dir_reader import SENSOR_CONFIGURATIONS


def height_filtration_from_top(value):
    return -value


def heigt_filtration_from_bottom(value):
    return value


def pershom_of_timeseries(timeseries, filtration):
    assert isinstance(timeseries, numpy.ndarray)
    assert timeseries.ndim == 1

    timeseries = timeseries.tolist()
    toplices = []
    filt_values = []

    for i in range(len(timeseries) - 1):
        # append vertex
        toplices.append((i,))
        filt_values.append(filtration(timeseries[i]))

        # append edge
        toplices.append((i, i + 1))
        filt_values.append(max(filtration(x) for x in (timeseries[i], timeseries[i + 1])))

    # append last vertex, which we do not get in loop
    toplices.append((len(timeseries) - 1,))
    filt_values.append(filtration(timeseries[len(timeseries) - 1]))

    return pershombox.toplex_persistence_diagrams(toplices=toplices,
                                                  filtration_values=filt_values,
                                                  deessentialize=True)


def job(args):
    index, data, meta = args
    dgms = defaultdict(list)
    for i_sensor in range(data.shape[1]):
        signal = data[:, i_sensor]
        signal = (signal - signal.mean()) / signal.std()

        dgms['top'].append(pershom_of_timeseries(signal, height_filtration_from_top)[0])
        dgms['bottom'].append(pershom_of_timeseries(signal, heigt_filtration_from_bottom)[0])

    return {'index': index,
            'dgms': dgms,
            'meta': meta}


def job_arg_iter(data_reader):
    assert isinstance(data_reader, SciNe01DataDirReader)
    for index in range(len(data_reader)):
        x, meta = data_reader[index]
        yield index, x, meta


read_me_txt = \
"""'data': access <index>/<filtration>/<sensor> \n'target': target[i] = label of 'data'[i]"""


def run():
    raw_data_dir = data_raw_path.joinpath('sciNe01_eeg')
    output_dir = data_generated_path.joinpath('sciNe01_eeg_pershom_bottom_top_filtration.h5')

    data_reader = SciNe01DataDirReader(raw_data_dir)

    progress = SimpleProgressCounter(len(data_reader))
    progress.display()
    n_cores = min(multiprocessing.cpu_count() - 1, 10)

    with h5py.File(output_dir, 'w') as h5file:

        grp_data = h5file.create_group('data')
        ds_target = h5file.create_dataset('target',
                                          dtype='i8',
                                          shape=(len(data_reader),))

        ds_group = h5file.create_dataset('group',
                                         dtype='i8',
                                         shape=(len(data_reader),))

        ds_run = h5file.create_dataset('run',
                                       dtype='i8',
                                       shape=(len(data_reader),))

        ds_sub_run = h5file.create_dataset('sub_run',
                                           dtype='i8',
                                           shape=(len(data_reader),))

        grp_sensor_cfg = h5file.create_group('sensor_configurations')
        for k, v in SENSOR_CONFIGURATIONS.items():
            grp_sensor_cfg.create_dataset(k, data=v, dtype='i8')

        ds_int_to_str_label = h5file.create_dataset('label_int_2_str',
                                                  (len(LABEL_IDS),),
                                                  dtype=h5py.special_dtype(vlen=str))
        for l_int, l_str in enumerate(LABEL_IDS):
            ds_int_to_str_label[l_int] = l_str

        ds_int_to_str_group = h5file.create_dataset('group_int_to_str',
                                                  (len(GROUP_IDS),),
                                                  dtype=h5py.special_dtype(vlen=str))
        for g_int, g_str in enumerate(GROUP_IDS):
            ds_int_to_str_group[g_int] = g_str

        h5file.attrs['readme'] = read_me_txt

        with multiprocessing.Pool(n_cores) as p:

            for ret_val in p.imap_unordered(job, job_arg_iter(data_reader)):
                index = ret_val['index']
                dgms = ret_val['dgms']
                meta = ret_val['meta']

                ds_target[index] = int_label_from_str_label(meta['label'])
                ds_group[index] = int_group_from_str_group((meta['group']))
                ds_run[index] = meta['run']
                ds_sub_run[index] = meta['sub_run']

                grp_index = grp_data.create_group(str(index))

                for filt_name, dgm_list in dgms.items():
                    grp_id_filt = grp_index.create_group(filt_name)

                    for i_sensor, dgm in enumerate(dgm_list):
                        dgm = np.array(dgm, dtype=np.float32)
                        grp_id_filt.create_dataset(str(i_sensor), data=dgm)

                progress.trigger_progress()
