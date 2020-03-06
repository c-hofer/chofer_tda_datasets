from .utils.h5py_dataset import Hdf5SupervisedDatasetOneFile


class SciNe01EEGBottomTopFiltration(Hdf5SupervisedDatasetOneFile):
    file_name = 'sciNe01_eeg_pershom_bottom_top_filtration.h5'

    @property
    def sensor_configurations(self):
        grp = self._h5py_file['sensor_configurations']
        return {k: v[()] for k, v in grp.items()}


class SciNe01EEGRawSignal(Hdf5SupervisedDatasetOneFile):
    file_name = 'sciNe01_eeg_raw_signal.h5'

    @property
    def sensor_configurations(self):
        grp = self._h5py_file['sensor_configurations']
        return {k: v[()] for k, v in grp.items()}


