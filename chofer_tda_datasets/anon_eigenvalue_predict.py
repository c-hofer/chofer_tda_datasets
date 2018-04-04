from .utils.h5py_dataset import Hdf5SupervisedDatasetOneFile


class AnonEigenvaluePredict(Hdf5SupervisedDatasetOneFile):
    file_name = 'anon_eigenvalue_predict_pershom_degree_filtration.h5'


class AnonEigenvaluePredictSmall(Hdf5SupervisedDatasetOneFile):
    file_name = 'anon_eigenvalue_predict_small_pershom_degree_filtration.h5'
