from torch.utils.data import DataLoader

from src.dataset import MyDataset

dataset = MyDataset(src_path='../data/zh-en/en.txt',
          trg_path='../data/zh-en/zh.txt',
          src_lang='en_core_web_sm',
          trg_lang='zh_core_web_sm')


data_loader = DataLoader(dataset, shuffle=True, batch_size=5, collate_fn=dataset.collate_fn)

for batch in data_loader:
    print(batch)
    break
