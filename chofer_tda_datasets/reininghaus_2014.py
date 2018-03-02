from .utils.h5py_dataset import Hdf5SupervisedDatasetOneFile


class Reininghaus2014ShrecReal(Hdf5SupervisedDatasetOneFile):
    file_name = 'reininghaus_2014_shrec_real.h5'


class Reininghaus2014ShrecSynthetic(Hdf5SupervisedDatasetOneFile):
    file_name = 'reininghaus_2014_shrec_synthetic.h5'
