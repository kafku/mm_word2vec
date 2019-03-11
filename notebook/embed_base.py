# coding: utf-8

import os
import json
import numpy as np
from scipy import stats
from gensim.models.keyedvectors import KeyedVectors, Vocab

class WordEmbedBase(object):
    def _tokens2idx(self, arr):
        return np.array([self.word_dict[x] for x in arr])

    def _squash_arrays(self, *arr):
        squashed_arr = []
        for x in arr:
            squashed_arr.append(self._squash(x))

        if len(squashed_arr) == 1:
            return squashed_arr[0]
        else:
            return tuple(squashed_arr)

    def evaluate_word_pairs(self, word_table, case_insensitive=True, dummy4unkown=False,
                            word1_col='word1', word2_col='word2', score_col='score'):
        '''word similarity task'''
        # TODO: do something for case_insesitive

        ok_vocab = self.word_dict
        similarity_model = []
        similarity_gold = []
        num_oov = 0

        if case_insensitive:
            pass
        else:
            pass

        for index, row in word_table.iterrows():
            word1 = row[word1_col]
            word2 = row[word2_col]
            # TODO: special processing for case_sensitve
            if word1 in ok_vocab and word2 in ok_vocab:
                similarity_model.append(self.wv.similarity(word1, word2))
                similarity_gold.append(row[score_col])
            else:
                num_oov += 1
                if dummy4unkown:
                    similarity_model.append(0.0)
                    similarity_gold.append(row[score_col])
                else:
                    # TODO: print something?
                    continue

        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unkown:
            oov_ratio = num_oov / len(similarity_gold) * 100
        else:
            oov_ratio = num_oov / (len(similarity_gold) + num_oov) * 100

        return pearson, spearman, oov_ratio

    def _set_keyedvector(self, attrname, keys, dim, vec=None):
        keyed_vec = KeyedVectors(dim)
        dummy_max_count = len(keys) + 1
        for i, key in enumerate(keys):
            key = str(key)
            keyed_vec.vocab[key] = Vocab(index=i, count=dummy_max_count - i) # dummy count
            keyed_vec.index2word.append(key)

        if vec is not None:
            keyed_vec.vectors = vec
            keyed_vec.init_sims()

        setattr(self, attrname, keyed_vec)

    def _set_keyedvector_val(self, attrname, vec):
        self[attrname].vectors = vec
        self[attrname].init_sims()

    def _save_meta_hook(self, model_meta):
        return model_meta

    def _save_np_params(self, dir_path, param_list=[]):
        if len(param_list) == 0:
            return

        params_to_save = dict(
            (param, getattr(self, param)) for param in param_list)
        np.savez(os.path.join(dir_path, 'params.npz'), **params_to_save)

    def save_model(self, dir_path, **kwargs):
        try:
            os.mkdir(dir_path)
        except FileExistsError as e:
            print('%s already exists'%e.filename)
            raise

        model_meta = {
            'model': type(self).__name__,
            'init_param': {
                'vocab_size': self.vocab_size,
                'window_size': self.window_size,
                'dim': self.dim
            },
            'non_init_param':{},
            'note': kwargs
        }
        model_meta = self._save_meta_hook(model_meta)
        with open(os.path.join(dir_path, 'model_meta.json'), 'w') as f:
            json.dump(model_meta, f, ensure_ascii=False,
                      indent=4, sort_keys=True, separators=(',', ': '))

        self.wv.save_word2vec_format(os.path.join(dir_path, 'word_vec.bin'), binary=True)

    @classmethod
    def load_model(cls, dir_path):
        with open(os.path.join(dir_path, 'model_meta.json'), 'r') as f:
            model_meta = json.load(f)

        assert cls.__name__ == model_meta['model']

        # create instance by constructor of cls
        model = cls(**model_meta['init_param'])

        for key, val in model_meta['non_init_param'].items():
            setattr(model, key, val)

        word2vec_model = KeyedVectors.load_word2vec_format(
            os.path.join(dir_path, 'word_vec.bin'), binary=True)
        model.wv = word2vec_model.wv

        try:
            with np.load(os.path.join(dir_path, 'params.npz'), mmap_mode='r') as data:
                for key, val in data.items():
                    setattr(model, key, val)
        except IOError as e:
            print('Failed to load %s.'%e.filename)

        return model


class SentEmbedBase(WordEmbedBase):
    pass
