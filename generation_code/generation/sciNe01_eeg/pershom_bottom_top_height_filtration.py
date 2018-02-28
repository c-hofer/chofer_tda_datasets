import multiprocessing

import h5py
import numpy
import numpy as np
import pershombox

from collections import defaultdict
from .data_dir_reader import SciNe01DataDirReader
from ..path_config import data_raw_path, data_generated_path
from ..utils.gui import SimpleProgressCounter


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
        #append vertex
        toplices.append((i,))
        filt_values.append(filtration(timeseries[i]))

        toplices.append((i, i + 1))
        filt_values.append(max(filtration(x) for x in (timeseries[i], timeseries[i + 1])))

    toplices.append((len(timeseries) - 1,))
    filt_values.append(filtration(timeseries[len(timeseries) - 1]))

    return pershombox.toplex_persistence_diagrams(toplices=toplices,
                                                  filtration_values=filt_values,
                                                  deessentialize=True)


def job(args):
    id, data, label = args
    dgms = defaultdict(list)
    for i_sensor in range(data.shape[1]):
        signal = data[:, i_sensor]
        signal = (signal - signal.mean()) / signal.std()

        dgms['top'].append(pershom_of_timeseries(signal, height_filtration_from_top)[0])
        dgms['bottom'].append(pershom_of_timeseries(signal, heigt_filtration_from_bottom)[0])

    return {'id': id,
            'dgms': dgms,
            'label': label}


def job_arg_iter(data_reader):
    assert isinstance(data_reader, SciNe01DataDirReader)
    for id in range(len(data_reader)):
        x, y = data_reader[id]
        yield id, x, y


#TODO add meta info strings
#TODO add group information
def run():
    raw_data_dir = data_raw_path.joinpath('sciNe01_eeg')
    output_dir = data_generated_path.joinpath('sciNe01_eeg_pershom_bottom_top_filtration.h5')

    data_reader = SciNe01DataDirReader(raw_data_dir, int_labels=True)

    progress = SimpleProgressCounter(len(data_reader))
    progress.display()
    n_cores = min(multiprocessing.cpu_count() - 1, 10)

    with h5py.File(output_dir, 'w') as h5file:

        grp_data = h5file.create_group('data')
        ds_target = h5file.create_dataset('target',
                                          dtype='i8',
                                          shape=(len(data_reader),))

        with multiprocessing.Pool(n_cores) as p:


            i = 0 #TODO remove


            for ret_val in p.imap_unordered(job, job_arg_iter(data_reader)):
                id = ret_val['id']
                dgms = ret_val['dgms']
                label = ret_val['label']

                ds_target[id] = label
                grp_id = grp_data.create_group(str(id))

                for filt_name, dgm_list in dgms.items():
                    grp_id_filt = grp_id.create_group(filt_name)

                    for i_sensor, dgm in enumerate(dgm_list):
                        dgm = np.array(dgm, dtype=np.float32)
                        grp_id_filt.create_dataset(str(i_sensor), data=dgm)

                progress.trigger_progress()

                i += 1 #TODO remove
                if i == 20: #TODO remove
                    break #TODO remove


if __name__ == "__main__":
    run()