import random
import sys
import glob
from typing import List
import numpy as np


class Vocabulary(object):
    """
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    """

    def __init__(self, filename, validate_file=False):
        """
        filename = the vocabulary file.
        It is a flat text file with one (normalized) token per line.
        In addition, the file should also contain the special tokens
        <S>, </S>, <UNK> (case sensitive).
        """
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """
        Convert a list of ids to a sentence, with space inserted.
        """
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        """
        Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately.
        """

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class CharsVocabulary(Vocabulary):
    """
    Vocabulary containing character-level and word level information.

    This function has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
    begin word, end word and padding.
    """

    def __init__(self, word_fn, char_fn, max_word_length, **kwargs):
        """
        filename = the vocabulary file.
        It is a flat text file with one (normalized) token per line.
        In addition, the file should also contain the special tokens
        <S>, </S>, <UNK> (case sensitive).
        """
        super(CharsVocabulary, self).__init__(word_fn, **kwargs)
        self._max_word_length = max_word_length

        self._id_to_char = []
        self._char_to_id = {}

        self.bow_char = -1
        self.eow_char = -1
        self.unk_char = -1

        idx = 0
        with open(char_fn) as f:
            for line in f:
                char_name = line.strip()
                if char_name == '<S>':
                    self.bow_char = idx
                elif char_name == '</S>':
                    self.eow_char = idx
                elif char_name == '<UNK>':
                    self.unk_char = idx

                self._id_to_char.append(char_name)
                self._char_to_id[char_name] = idx
                idx += 1

        assert self.bow_char > -1 and self.eow_char > -1 and self.unk_char > -1
        self.pad_char = idx  # <padding>

        num_words = len(self._id_to_word)
        self._word_char_ids = np.zeros([num_words, max_word_length],
                                       dtype=np.int32)

        # the representation of the begin/end of sentence characters
        def _make_bos_eos(_char_id):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = _char_id
            r[2] = self.eow_char
            return r

        self.bos_chars = _make_bos_eos(self.bos)
        self.eos_chars = _make_bos_eos(self.eos)

        # Pre-compute ids of the words in the vocab
        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)
        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars

    @property
    def char_size(self):
        return self.pad_char + 1

    def char_to_id(self, char):
        if char in self._char_to_id:
            return self._char_to_id[char]
        return self.unk

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char
        code[0] = self.bow_char
        word_trimmed = word[:self.max_word_length - 2]
        for k, char in enumerate(word_trimmed, start=1):
            code[k] = self.char_to_id(char)
        code[k + 1] = self.eow_char
        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True):
        """
        Encode the sentence as a white space delimited string of tokens.
        """
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


# for training
def _get_batch(generator, batch_size, num_steps, max_word_length):
    """
    Read batches of input.
    """
    cur_stream = [None] * batch_size

    no_more_data = False
    while True:
        if no_more_data:
            # There is no more data.
            # Note: this will not return data for the incomplete batch
            break

        token_inputs = np.zeros([batch_size, num_steps], np.int32)
        if max_word_length is not None:
            char_inputs = np.zeros([batch_size, num_steps, max_word_length],
                                   np.int32)
        else:
            char_inputs = None
        targets = np.zeros([batch_size, num_steps], np.int32)

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        # list of (token_id, char_id) tuples
                        # token_id: 1D: n_tokens
                        # char_id: 1D: n_tokens, 2D: max_characters_per_token
                        cur_stream[i] = list(next(generator))
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                # Add stream to batch
                token_inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                if max_word_length is not None:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many + 1]

                # Update for the next iteration
                cur_pos = next_pos

                cur_stream[i][0] = cur_stream[i][0][how_many:]
                if max_word_length is not None:
                    cur_stream[i][1] = cur_stream[i][1][how_many:]

        """
        if no_more_data:
            # There is no more data.
            # Note: this will not return data for the incomplete batch
            break
        """

        X = {'token_ids': token_inputs,
             'tokens_characters': char_inputs,
             'next_token_id': targets}

        yield X


class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files.
    Each file contains one sentence per line.
    Each sentence is pre-tokenized and white space joined.
    """

    def __init__(self,
                 file_path,
                 vocab,
                 reverse=False,
                 test=False,
                 shuffle_on_load=False):
        """
        file_path = a glob string that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        shuffle_on_load = if True, then shuffle the sentences after loading.
        """
        self._vocab = vocab
        self.n_tokens = 0
        self.file_path = file_path
        self.all_shards = glob.glob(file_path)
        print('Found %d shards at %s' % (len(self.all_shards), file_path))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        self._ids = self._load_random_shard()

    def reload_all_shards(self):
        self.all_shards = glob.glob(self.file_path)

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self.all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        """
        Randomly select a file and read it.

        Returns:
            list of (id, char_id) tuples.
        """
        if self._test:
            if len(self.all_shards) == 0:
                # we've loaded all the data
                # this will propagate up to the generator in get_batch
                # and stop iterating
                raise StopIteration
            else:
                shard_name = self.all_shards.pop()
        else:
            # just pick a random shard
            shard_name = self._choose_random_shard()

        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _load_shard(self, shard_name):
        """
        Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        sys.stdout.write('\nLoading data from: %s\n' % shard_name)
        sys.stdout.flush()

        ##################
        # Load sentences #
        ##################
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        #####################
        # Shuffle sentences #
        #####################
        if self._shuffle_on_load:
            random.shuffle(sentences)

        ###########################
        # Convert str to token id #
        ###########################
        token_ids = [self.vocab.encode(sentence, self._reverse)
                     for sentence in sentences]

        ##########################
        # Convert str to char id #
        ##########################
        if self._use_char_inputs:
            char_ids = [self.vocab.encode_chars(sentence, self._reverse)
                        for sentence in sentences]
        else:
            char_ids = [None] * len(token_ids)

        ################
        # Count tokens #
        ################
        n_tokens = 0
        for ids_i in token_ids:
            n_tokens += len(ids_i)
        self.n_tokens = n_tokens

        sys.stdout.write('Loaded %d sentences  %d tokens.\n' % (len(token_ids), n_tokens))
        sys.stdout.write('Finished loading\n')
        sys.stdout.flush()
        return list(zip(token_ids, char_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(generator=self.get_sentence(),
                            batch_size=batch_size,
                            num_steps=num_steps,
                            max_word_length=self.max_word_length):
            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, max_word_length) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X

    @property
    def vocab(self):
        return self._vocab


class BidirectionalLMDataset(object):
    def __init__(self, file_path, vocab, test=False, shuffle_on_load=False):
        """
        bidirectional version of LMDataset
        """
        self._vocab = vocab
        self.data_forward = LMDataset(
            file_path, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        self.data_reverse = LMDataset(
            file_path, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)

    def reload_all_shards(self):
        if len(self.data_forward.all_shards) == 0:
            self.data_forward.reload_all_shards()
            self.data_reverse.reload_all_shards()

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self.data_forward.max_word_length

        for X, Xr in zip(
                _get_batch(self.data_forward.get_sentence(),
                           batch_size,
                           num_steps,
                           max_word_length),
                _get_batch(self.data_reverse.get_sentence(),
                           batch_size,
                           num_steps,
                           max_word_length)
        ):
            for k, v in Xr.items():
                X[k + '_reverse'] = v

            yield X

    @property
    def vocab(self):
        return self._vocab


class Batcher(object):
    """
    Batch sentences of tokenized text into character id matrices.
    """

    def __init__(self, word_file, char_file=None, max_word_length=None):
        """
        lm_vocab_file = the language model vocabulary file
                        (one line per token)
        max_token_length = the maximum number of characters in each token
        """
        self._lm_vocab = CharsVocabulary(word_file,
                                         char_file,
                                         max_word_length,
                                         validate_file=True
                                         )
        self._max_token_length = max_word_length

    def batch_sentences(self, sentences: List[List[str]]):
        """
        Batch the sentences as character ids.
        Each sentence is a list of tokens without <s> or </s>,
            e.g. [['The', 'first', 'sentence', '.'], ['Second', '.']]
        """
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(sent,
                                                                split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids
