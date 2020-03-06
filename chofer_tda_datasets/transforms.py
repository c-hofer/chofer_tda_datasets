import h5py


class Hdf5GroupListSelector:
    def __init__(self, keys: [str]):
        self.keys = keys

    def __call__(self, data_grp)->[]:
        assert isinstance(data_grp, h5py.Group)
        return [data_grp[key][()] for key in self.keys]


class Hdf5GroupToDict:
    def __init__(self):
        pass

    def __call__(self, data_grp: h5py.Group):
        assert isinstance(data_grp, (h5py.Group, h5py.Dataset))
        if isinstance(data_grp, h5py.Dataset):
            return data_grp[()]
        else:
            return {k: self(v) for k, v in data_grp.items()}


class Hdf5GroupToDictSelector:
    def __init__(self, key_selection: {str}):
        self.key_selection = key_selection

    def __select(self, data_grp, selection):
        if isinstance(selection, dict):
            return {k: self.__select(data_grp[k], v) for k, v in selection.items()}
        else:
            assert isinstance(selection, (list, str))
            return {k: data_grp[k][()] for k in selection}

    def __call__(self, data_grp: h5py.Group):
        assert isinstance(data_grp, h5py.Group)
        return self.__select(data_grp, self.key_selection)
