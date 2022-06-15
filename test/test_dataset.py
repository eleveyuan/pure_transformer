from torch.utils.data import DataLoader

from src.dataset import read_data, MyDataset, collate_fn


src_data, trg_data, src_vocab_size, trg_vocab_size = read_data(src_path='../data/zh-en/en.txt',
          trg_path='../data/zh-en/zh.txt',
          src_lang='en_core_web_sm',
          trg_lang='zh_core_web_sm')


data_loader = DataLoader(MyDataset(src_data, trg_data), shuffle=True, batch_size=5, collate_fn=collate_fn)

for batch in data_loader:
    print(batch)
