# elmo-japanese
Tensorflow implementation of bidirectional language models (biLM) used to compute ELMo representations
from ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365).

This codebase is based on [bilm-tf](https://github.com/allenai/bilm-tf) and deals with Japanese.

This repository supports both training biLMs and using pre-trained models for prediction.

Citation:

```
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```
## Installation
- CPU
```
conda create -n bilm-jp python=3.6 anaconda
source activate bilm-jp
pip install tensorflow==1.10 h5py
git clone https://github.com/cl-tohoku/bilm-jp.git
```
- GPU
```
conda create -n bilm-jp python=3.6 anaconda
source activate bilm-jp
pip install tensorflow-gpu==1.10 h5py
git clone https://github.com/cl-tohoku/bilm-jp.git
```

## Getting started
- Training a biLM
```
python src/run_train.py \
    --option_file data/config.json \
    --save_dir checkpoint \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --train_prefix data/sample.jp.wakati.txt
```

- Computing representations from the trained biLM

The following command outputs the ELMo representations (elmo.hdf5) for the text (sample.jp.wakati.txt) in the checkpoint directory (save_dir).

```
python src/run_elmo.py \
    --option_file checkpoint/options.json \
    --weight_file checkpoint/weight.hdf5 \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --data_file data/sample.jp.wakati.txt \
    --output_file elmo.hdf5
```

The following command prints out the information of the elmo.hdf5, such as the number of sentences, words and dimensions.

```
python scripts/view_hdf5.py elmo.hdf5
```


## Computing sentence representations
- Save sentence-level ELMo representations
```
python src/run_elmo.py \
    --option_file checkpoint/options.json \
    --weight_file checkpoint/weight.hdf5 \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --data_file data/sample.jp.wakati.txt \
    --output_file elmo.hdf5 \
    --sent_vec
```

- View sentence similarities
```
python scripts/view_sent_sim.py \
    --data data/sample.jp.wakati.txt \
    --elmo elmo.hdf5
```


## Training a biLM on a new corpus
- Making a token vocab file
```
python src/make_vocab_file.py \
    --input_fn data/sample.jp.wakati.txt \
    --output_fn data/vocab.sample.jp.wakati.txt
```

- Making a character vocab file
```
python scripts/space_split.py \
    --input_fn data/sample.jp.wakati.txt \
    --output_fn data/sample.jp.space.txt
```
```
python scripts/make_vocab.py \
    --input_fn data/sample.jp.space.txt \
    --output_fn data/vocab.sample.jp.space.txt
```

- Training a biLM
```
python src/run_train.py \\
    --train_prefix data/sample.jp.wakati.txt \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --config_file data/config.json
    --save_dir checkpoint
```

- Retraining the trained biLM
```
python src/run_train.py \
    --train_prefix data/sample.jp.wakati.txt \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --save_dir checkpoint \
    --restart
```

- Computing representations from the biLM

```
python src/run_elmo.py \
    --test_prefix data/sample.jp.wakati.txt \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --save_dir checkpoint
```

- Testing
```
python src/run_test.py \
    --test_prefix data/sample.jp.wakati.txt \
    --word_file data/vocab.sample.jp.wakati.txt \
    --char_file data/vocab.sample.jp.space.txt \
    --save_dir checkpoint
```

- Pre-trained model
* [checkpoint](https://drive.google.com/open?id=1kqobOypRhdUgpkDYZmXACWYamcP4acQf)
