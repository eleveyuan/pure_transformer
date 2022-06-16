import argparse
import re

import torch
from torch.autograd import Variable
import dill as pickle
from nltk.corpus import wordnet

from src.model import get_model
from src.beam import beam_search


def get_synonym(word, src_w2i):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if src_w2i[l.name()] != 0:
                return src_w2i[l.name()]
    return 0


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, dataset):
    model.eval()
    indexed = []
    # preprocess sentence
    src_w2i = dataset.src_w2i
    trg_w2i = dataset.trg_w2i
    trg_i2w = dataset.trg_i2w

    tokenize = dataset.get_tokenize(opt.src_lang)

    sentence = tokenize.tokenizer(sentence)
    for tok in sentence:
        if src_w2i[tok] != 0:
            indexed.append(src_w2i[tok])
        else:
            indexed.append(get_synonym(tok, src_w2i))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()

    sentence = beam_search(sentence, model, src_w2i, trg_w2i, trg_i2w, opt)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model, dataset):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, dataset).capitalize())

    return ' '.join(translated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', default=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', default='en_core_web_sm')
    parser.add_argument('-trg_lang', default='zh_core_web_sm')
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-floyd', action='store_true')

    opt = parser.parse_args()

    opt.device = 0 if torch.cuda.is_available() else -1

    assert opt.k > 0
    assert opt.max_len > 10

    dataset = None
    try:
        print("loading presaved fields...")
        dataset = pickle.load(open(f'weights/dataset.pkl', 'rb'))
    except:
        print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
        quit()

    model = get_model(opt, dataset.src_vocab_size, dataset.trg_vocab_size)

    while True:
        opt.text = input("Enter a sentence to translate (type 'q' to quit):\n")
        if opt.text == "q":
            break
        phrase = translate(opt, model, dataset)
        print('> ' + phrase + '\n')


if __name__ == '__main__':
    main()