Multimodal Skip-gram Model
========

This is an **unoffical** implementation of [multimodal Skip-gram model](http://www.aclweb.org/anthology/N15-1016). Forked from [Word2Vec in C++11](https://github.com/jdeng/word2vec).

## Prerequisites

- g++ >= 4.8.5
- Boost >= 1.53.0
- openblas >= 0.3.3
- HDF5 (library) >= 1.8.12

## Usage

```bash
# compile sources
$ make

# By -h option, you can find the details of its options
$ ./word2vec -h
Usage : ./word2vec [options] input_path
Allowed options:
  -h [ --help ]                         help.
  -m [ --mode ] arg (=train)            Mode train/test.
  -o [ --output ] arg (=./vectors.bin)  Output path.
  -d [ --dim ] arg (=300)               Dimensionality of word embedding.
  -w [ --window ] arg (=5)              Window size.
  -s [ --sample ] arg (=0.00100000005)  Subsampling probability.
  -c [ --min-count ] arg (=5)           The minimum frequency of words.
  -n [ --negative ] arg (=5)            The number of negative samples.
  -a [ --alpha ] arg (=0.0250000004)    The initial learning rate.
  -b [ --min-alpha ] arg (=9.99999975e-05)
                                        The minimum learning rate.
  -p [ --n_workers ] arg (=0)           The number of threads
  -f [ --format ] arg (=bin)            Output file format: bin/text
  -i [ --iteration ] arg (=5)           The number of iterations
  -M [ --method ] arg (=HS)             Methos: HierarchicalSoftmax(HS)/Negativ
                                        eSampling(NS)
  -I [ --multimodal-input ] arg         Path to multimodal feature file
  --input_path arg                      Path to input file
```

With the ` --multimodal-input` option, it works as multimodal skip-gram model, otherwise it just the same as the ordinary word2vec.

## Reference

- [Combining Language and Vision with a Multimodal Skip-gram Model](https://aclanthology.info/papers/N15-1016/n15-1016)

