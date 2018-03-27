from .utils.h5py_dataset import Hdf5SupervisedDatasetOneFile


class Reddit5kJmlr(Hdf5SupervisedDatasetOneFile):
    file_name = 'reddit_5k_jmlr.h5'


class Reddit12kJmlr(Hdf5SupervisedDatasetOneFile):
    file_name = 'reddit_12k_jmlr.h5'