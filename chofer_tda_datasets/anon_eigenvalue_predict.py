from .utils.h5py_dataset import Hdf5SupervisedDatasetOneFile


class Anon10kEigenvaluePredict(Hdf5SupervisedDatasetOneFile):
    file_name = 'anon_10k_eigenvalue_predict_pershom_degree_filtration.h5'


class Anon1kEigenvaluePredict(Hdf5SupervisedDatasetOneFile):
    file_name = 'anon_1k_eigenvalue_predict_small_pershom_degree_filtration.h5'
