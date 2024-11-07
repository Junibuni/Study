import torch
from torchtext.datasets import Multi30k
from torchtext.transforms import Sequential, VocabTransform, Truncate, AddToken
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        tokenizer_src = get_tokenizer('spacy', language='en_core_web_sm')
        tokenizer_tgt = get_tokenizer('spacy', language='es_core_news_sm')

        self.data = list(Multi30k(split=split, language_pair=('en', 'es')))
        self.vocab_src = build_vocab_from_iterator((tokenizer_src(text) for text, _ in self.data), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        self.vocab_tgt = build_vocab_from_iterator((tokenizer_tgt(text) for _, text in self.data), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        
        self.transforms_src = Sequential(VocabTransform(self.vocab_src), AddToken('<bos>'), AddToken('<eos>'))
        self.transforms_tgt = Sequential(VocabTransform(self.vocab_tgt), AddToken('<bos>'), AddToken('<eos>'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tensor = torch.tensor(self.transforms_src(src))
        tgt_tensor = torch.tensor(self.transforms_tgt(tgt))
        return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=1, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    return src_batch, tgt_batch

# Example usage
if __name__ == "__main__":
    dataset = TranslationDataset(split='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    for src, tgt in data_loader:
        print(src.shape, tgt.shape)
        break
