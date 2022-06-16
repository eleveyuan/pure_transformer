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


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0).clone().detach()
    if torch.cuda.is_available():
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg, src_pad=0, trg_pad=0):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


class MyDataset(Dataset):
    def __init__(self, src_path='../data/zh-en/en.txt',
                trg_path='../data/zh-en/zh.txt',
                src_lang='en_core_web_sm',
                trg_lang='zh_core_web_sm'):
        super(MyDataset, self).__init__()
        src_data = open(src_path, encoding='utf8').read().strip().split('\n')
        trg_data = open(trg_path, encoding='utf8').read().strip().split('\n')

        self.src_data2ids, self.src_w2i, self.src_i2w, self.src_vocab_size = self.build_vocab(src_data, src_lang)
        self.trg_data2ids, self.trg_w2i, self.trg_i2w, self.trg_vocab_size = self.build_vocab(trg_data, trg_lang)

    def get_tokenize(self, lang):
        # you can use another tokenize tool
        spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'zh_core_web_sm']
        spacy_langs_str = '[' + ','.join(spacy_langs) + ']'
        if lang not in spacy_langs:
            print('invalid language: ' + lang + 'supported languages : ' + spacy_langs_str)

        return Tokenize(lang)

    def build_vocab(self, data, lang):
        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for i, st in enumerate(special_tokens):
            i2w[i] = st
            w2i[st] = i
        lines = []
        tokenize = self.get_tokenize(lang)
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

    def __getitem__(self, idx):
        return [self.src_data2ids[idx], self.trg_data2ids[idx]]

    def __len__(self):
        return len(self.src_data2ids)

    def collate_fn(self, batch):
        # self-defined collate_fn for DataLoader
        # batch like: [[[138, 142, 116, 819, 2813, 437, 667, 5], [268, 462, 1093, 137, 4265, 1025, 901, 5]], [],...]
        src_ = [src_trg[0] for src_trg in batch]
        trg_ = [src_trg[1] for src_trg in batch]
        src_tokens = [[self.src_w2i['<sos>']] + token_ + [self.src_w2i['<eos>']] for token_ in src_]
        tgt_tokens = [[self.trg_w2i['<sos>']] + token_ + [self.trg_w2i['<eos>']] for token_ in trg_]

        # then you should padding sequence
        src_batch = pad_sequence([torch.LongTensor(line) for line in src_tokens],
                                 batch_first=True, padding_value=self.src_w2i['<pad>'])
        trg_batch = pad_sequence([torch.LongTensor(line) for line in tgt_tokens],
                                 batch_first=True, padding_value=self.trg_w2i['<pad>'])
        
        # we input all words except the last, as it is using each word to predict the next
        trg_batch_y = trg_batch[:, 1:]
        trg_batch = trg_batch[:, :-1]

        # for NMT task in transformer, you mask some sequence information
        src_mask, trg_mask = create_masks(src_batch, trg_batch, self.src_w2i['<pad>'], self.trg_w2i['<pad>'])
        return src_batch, trg_batch, trg_batch_y, src_mask, trg_mask
