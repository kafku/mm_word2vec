# coding: utf-8

import itertools
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from proxy import Proxy


class NamedArrBase(object):
    pass


class NamedArray(Proxy, NamedArrBase):
    # FIXME: issparse(NamedArray(csr_matrix)) -> False
    # FIXME: error when NamedArray(csr_matrix) @ NamedArray(csr_matrix)
    def __init__(self, data, axis_names):
        super().__init__(data)
        self._names = []
        assert len(axis_names) == len(data.shape)
        for names, dim in zip(axis_names, data.shape):
            assert names is None or len(names) == dim
            self._names.append(pd.Index(names) if names else None)

    def issparse(self):
        return sparse.issparse(object.__getattribute__(self, "_obj"))

    def __getitem__(self, key):
        key, new_names = self._convert_key(key)
        sliced_data = getattr(object.__getattribute__(self, "_obj"), "__getitem__")(key)
        return NamedArray(sliced_data, axis_names=new_names)

    def _convert_key(self, key):
        # FIXME: what if key is a single integer? -> no names?
        # FIXME: error when key includes str x['row1', ] -> vector with no names?
        # FIXME: error when 1-dim array x['a']
        # FIXME: error when x[['row1', 'row2'], ] -> no colnames
        new_key = []
        new_names = []
        if not isinstance(key, tuple):
            key = key,
        for i, idx in enumerate(key):
            if idx is None:
                new_key.append(None)
                new_names.append(None)
            elif isinstance(idx, int):
                new_key.append(idx)
                if self._names[i] is None:
                    new_names.append(None)
                else:
                    new_names.append([self._names[i][idx]])
            elif isinstance(idx, slice):
                new_key.append(idx)
                if self._names[i] is None:
                    new_names.append(None)
                else:
                    new_names.append(self._names[i][idx].tolist())
            elif isinstance(idx, (np.ndarray, list)):
                convert = lambda x: x if isinstance(x, int) else self._names[i].get_loc(x)
                if isinstance(idx, list):
                    new_key.append([convert(x) for x in idx])
                elif idx.dtype == np.bool:
                    # append asis when bool ndarray
                    new_key.append(idx)
                else:
                    # convert() when integer or string ndarray
                    for elem in np.nditer(idx, op_flags=['readwrite']):
                        elem[...] = convert(np.asscalar(elem))
                    new_key.append(idx.astype(np.int32))

                if self._names[i] is None:
                    new_names.append(None)
                else:
                    new_names.append(self._names[i][np.ravel(new_key[-1])].tolist())
            else:
                raise ValueError() # FIXME
        return tuple(new_key), new_names

    @property
    def names(self):
        return tuple(names.tolist() if names is not None else None
                     for names in self._names) # FIXME: slow?

    @names.setter
    def names(self, axes=0):
        # TODO: check length and update names
        raise NotImplementedError()

    def to_hdf(self, path, group=None, mode='w'):
        with h5py.File(path, mode=mode) as h5_file:
            if group is None:
                h5_obj = h5_file
            else:
                h5_obj = h5_file.create_group(group)
            for i, names in enumerate(self._names):
                if names is None:
                    h5_obj.create_dataset("name_%d"%i, dtype='|S2')
                else:
                    h5_obj.create_dataset("name_%d"%i,
                                          data=[x.encode('utf-8') for x in names])
            if isinstance(self._obj, np.ndarray):
                h5_obj.attrs['type'] = 'ndarray'.encode('utf-8')
                h5_obj.create_dataset("arr", data=self._obj)
            elif isinstance(self._obj, (sparse.csr_matrix, sparse.csc_matrix)):
                h5_obj.attrs['type'] = type(self._obj).__name__.encode('utf-8')
                h5_obj.attrs['shape'] = self._obj.shape
                h5_obj.create_dataset('data', data=self._obj.data)
                h5_obj.create_dataset('indptr', data=self._obj.indptr)
                h5_obj.create_dataset('indices', data=self._obj.indices)

    @classmethod
    def load(cls, path, group=None):
        with h5py.File(path, mode='r') as h5_file:
            if group is None:
                h5_obj = h5_file
            else:
                h5_obj = h5_file[group]

            data_type = h5_obj.attrs['type'].decode('utf-8')
            arr = None
            if data_type == 'ndarray':
                arr = h5_obj['arr']
            elif data_type == 'csr_matrix' or data_type == 'csc_matrix':
                shape = h5_obj.attrs['shape']
                data = h5_obj['data']
                indptr = h5_obj['indptr']
                indices = h5_obj['indices']
                if data_type == 'csr_matrix':
                    arr = sparse.csr_matrix((data, indices, indptr), shape=shape)
                elif data_type == 'csc_matrix':
                    arr = sparse.csc_matrix((data, indices, indptr), shape=shape)

            names = []
            for i in range(len(arr.shape)):
                if isinstance(h5_obj['name_%d'%i], h5py.Empty):
                    names.append(None)
                else:
                    names.append([x.decode('utf-8') for x in h5_obj['name_%d'%i]])

        return NamedArray(arr, axis_names=names)

    _overrided_special_names = ["__getitem__"]

    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""

        def make_method(name):
            def method(self, *args, **kw):
                return getattr(object.__getattribute__(self, "_obj"), name)(*args, **kw)
            return method

        namespace = {}
        for name in cls._special_names:
            if name in cls._overrided_special_names:
                continue
            if hasattr(theclass, name):
                namespace[name] = make_method(name)
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)


def vstack(blocks, **kwargs):
    # TODO: check if all the array are NameArray
    sparse_stack = any(arr.issparse() for arr in blocks)
    if sparse_stack:
        stacked_arr = sparse.vstack([x._obj for x in blocks], **kwargs)
    else:
        stacked_arr = np.vstack([x._obj for x in blocks])
    new_names = []
    new_names.append([*itertools.chain.from_iterable(arr.names[0] for arr in blocks)])
    new_names.append(blocks[0].names[1])

    return NamedArray(stacked_arr, axis_names=new_names)


def remove_name(x):
    if isinstance(x, NamedArrBase):
        return x._obj
    else:
        return x
