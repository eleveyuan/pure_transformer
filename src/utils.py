import os

import torch
from torchtext import data
from torch.autograd import Variable
import numpy as np
import pandas as pd
import dill as pickle

from src.tokenize import tokenize


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()


def create_fields(opt):
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    spacy_langs_str = '[' + ','.join(spacy_langs) + ']'
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs_str)
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs_str)

    print("loading spacy tokenizers...")

    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(opt, SRC, TRG):
    print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)

    return train_iter


def get_len(train):
    for i, b in enumerate(train):
        pass

    return i
