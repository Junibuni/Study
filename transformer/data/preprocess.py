import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        self.tokenizer_src = get_tokenizer('spacy', language='en_core_web_sm')
        self.tokenizer_tgt = get_tokenizer('spacy', language='es_core_news_sm')

        self.data = list(Multi30k(split=split, language_pair=('en', 'de')))
        self.vocab_src = build_vocab_from_iterator(self.yield_tokens(self.data, index=0), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        self.vocab_tgt = build_vocab_from_iterator(self.yield_tokens(self.data, index=1), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        
        self.vocab_src.set_default_index(self.vocab_src['<unk>'])
        self.vocab_tgt.set_default_index(self.vocab_tgt['<unk>'])

    def yield_tokens(self, data_iter, index):
        for src, tgt in data_iter:
            tokens = self.tokenizer_src(src) if index == 0 else self.tokenizer_tgt(tgt)
            yield tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = ['<bos>'] + self.tokenizer_src(src) + ['<eos>']
        tgt_tokens = ['<bos>'] + self.tokenizer_tgt(tgt) + ['<eos>']
        src_indices = [self.vocab_src[token] for token in src_tokens]
        tgt_indices = [self.vocab_tgt[token] for token in tgt_tokens]
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=1, batch_first=True)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    return src_batch, tgt_batch

if __name__ == "__main__":
    dataset = TranslationDataset(split='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    for src, tgt in data_loader:
        print(src.shape, tgt.shape)
        break
