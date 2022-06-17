import os
import time
import pickle
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.model import get_model
from src.optim import CosineWithRestarts
from src.dataset import MyDataset


def train(model, dataset, opt):
    print("training model...")

    trg_pad = dataset.trg_w2i['<pad>']
    src_pad = dataset.src_w2i['<pad>']

    dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batchsize, collate_fn=dataset.collate_fn)
    iter_len = len(dataloader)

    if opt.SGDR:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=iter_len)

    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    for epoch in range(opt.epochs):
        total_loss = 0

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        counter = 0
        for src, trg, trg_y, src_mask, trg_mask in dataloader:
            # trg_input = trg[:, :-1]
            preds = model(src, trg, src_mask, trg_mask)
            # print(src, trg, src_mask, trg_mask)
            # ys = trg[:, :-1].contiguous().view(-1)
            ys = trg_y.contiguous().view(-1)
            # print(preds.view(-1, preds.size(-1)).size(), ys.size())
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
            loss.backward()
            opt.optimizer.step()

            if opt.SGDR:
                opt.sched.step()

            total_loss += loss.item()

            if (counter + 1) % opt.printevery == 0:
                p = int(100 * (counter + 1) / iter_len)
                avg_loss = total_loss / opt.printevery

                print("=>   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss))
                total_loss = 0

            if opt.checkpoint > 0 and ((time.time() - cptime) // 60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()

            counter += 1

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))

    if not os.path.exists('weights'):
        os.mkdir('weights')
    if not opt.load_weights:
        pickle.dump(dataset, open('weights/dataset.pkl', 'wb'))
        torch.save(model.state_dict(), 'weights/model_weights')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_path', default='./data/zh-en/en.txt')
    parser.add_argument('-trg_path', default='./data/zh-en/zh.txt')
    parser.add_argument('-src_lang', default='en_core_web_sm')
    parser.add_argument('-trg_lang', default='zh_core_web_sm')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=32)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights',action='store_true', default=False)
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-printevery', type=int, default=100)

    opt = parser.parse_args()

    opt.device = 0 if torch.cuda.is_available() else -1

    dataset = MyDataset(src_path=opt.src_path,
                        trg_path=opt.trg_path,
                        src_lang=opt.src_lang,
                        trg_lang=opt.trg_lang)
    model = get_model(opt, dataset.src_vocab_size, dataset.trg_vocab_size)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    if opt.checkpoint > 0:
        print(
            "model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    train(model, dataset, opt)


if __name__ == '__main__':
    main()
