import re
from collections import Counter, OrderedDict

import spacy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Tokenize(object):

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def build_vocab(data, tokenize):
    w2c = OrderedCounter()
    w2i = dict()
    i2w = dict()
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    for i, st in enumerate(special_tokens):
        i2w[i] = st
        w2i[st] = i
    lines = []
    for i, line in enumerate(data):
        words = tokenize.tokenizer(line)
        lines.append(words)
        w2c.update(words)
    for w, c in w2c.items():
        # maybe you shouled filter some low frequent word
        if w not in special_tokens:
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)
    data2ids = []
    for line in lines:
        data2ids.append([w2i[el] for el in line])
    return data2ids, w2i, i2w, len(w2i)


def read_data(src_path='../data/zh-en/en.txt',
              trg_path='../data/zh-en/zh.txt',
              src_lang='en_core_web_sm',
              trg_lang='zh_core_web_sm'):
    src_data = open(src_path).read().strip().split('\n')
    trg_data = open(trg_path).read().strip().split('\n')

    # you can use another tokenize tool
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'zh_core_web_sm']
    spacy_langs_str = '[' + ','.join(spacy_langs) + ']'
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + spacy_langs_str)
    if trg_lang not in spacy_langs:
        print('invalid trg language: ' + trg_lang + 'supported languages : ' + spacy_langs_str)

    print("loading spacy tokenizers...")
    t_src = Tokenize(src_lang)
    t_trg = Tokenize(trg_lang)

    src_data2ids, src_w2i, src_i2w, src_vocab_size = build_vocab(src_data, t_src)
    trg_data2ids, trg_w2i, trg_i2w, trg_vocab_size = build_vocab(trg_data, t_trg)

    return src_data2ids, trg_data2ids, src_vocab_size, trg_vocab_size

def create_masks():
    pass


def collate_fn(batch):
    # self-defined collate_fn for DataLoader
    # batch like this: [[[138, 142, 116, 819, 2813, 437, 667, 5], [268, 462, 1093, 137, 4265, 1025, 901, 5]], [],...]
    src_ = [src_trg[0] for src_trg in batch]
    trg_ = [src_trg[1] for src_trg in batch]
    src_tokens = [[2] + token_ + [3] for token_ in src_]
    tgt_tokens = [[2] + token_ + [3] for token_ in trg_]

    # then you should padding sequence
    batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                batch_first=True, padding_value=0)
    batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                batch_first=True, padding_value=0)

    # for NMT task in transformer, you mask some sequence information
    create_masks()


class MyDataset(Dataset):
    def __init__(self, src, trg):
        super(MyDataset, self).__init__()
        self.src = src
        self.trg = trg

    def __getitem__(self, idx):
        return [self.src[idx], self.trg[idx]]

    def __len__(self):
        return len(self.src)


