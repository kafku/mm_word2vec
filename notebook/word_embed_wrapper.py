# coding: utf-8

import pandas as pd
from embed_base import WordEmbedBase
import named_array as na


class WordEmbedWrapper(WordEmbedBase):
    def __init__(self, vocab, vec):
        assert len(vocab) == vec.shape[0]
        self.dim = vec.shape[1]
        self.wv = None
        self._set_keyedvector('wv', vocab, self.dim, vec=vec)


class MultimodalWordEmbedWrapper(WordEmbedWrapper):
    def __init__(self, vocab, vec, img_transform):
        assert callable(img_transform)
        self.mapped_image = None
        self.img_transform = img_transform
        super().__init__(vocab, vec)

    def map_image(self, images, image_ids=None):
        if isinstance(images, pd.DataFrame):
            image_ids = images.index.tolist()
            images = images.values
        elif isinstance(images, na.NamedArrBase):
            image_ids = images.names[0]

        self._set_keyedvector('mapped_image', image_ids, self.dim,
                              vec=self.img_transform(images))

    def most_similar(self, pos_word=[], neg_word=[], pos_img=[], neg_img=[],
                     target="word", topn=10):
        positive = []
        negative = []
        positive.extend(self.wv.word_vec(x, use_norm=True) for x in pos_word)
        positive.extend(self.mapped_image.word_vec(x, use_norm=True) for x in pos_img)
        negative.extend(self.wv.word_vec(x, use_norm=True) for x in neg_word)
        negative.extend(self.mapped_image.word_vec(x, use_norm=True) for x in neg_img)
        if target == "word":
            return self.wv.most_similar(positive=positive, negative=negative, topn=topn)
        elif target == "image":
            return self.mapped_image.most_similar(positive=positive, negative=negative, topn=topn)
        else:
            raise ValueError("invalid target. target must be one of word or image")
