import torch
import torch.nn as nn
import torch.nn.functional as F
from m3care.cs224n.highway import Highway

class ModelEmbeddings(nn.Module):
    def __init__(self, vocab, embedding_dim, out_channels, kernel_size, dropout_rate):
        super(ModelEmbeddings, self).__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_word_length = max([len(w) for w in vocab.id2word.values()])
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.char_vocab_dim = 127

        self.embedding = nn.Embedding(self.char_vocab_dim, self.embedding_dim, padding_idx=1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.char_conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(embedding_dim, kernel_size))
        # self.conv = nn.Conv1d(in_channels=embedding_dim,
        #                     out_channels=out_channels,
        #                     kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def expand_sentence(self, sent):
        sent = self.vocab.indices2words(sent.tolist())
        sent = [[ord(c) for c in s] for s in sent]
        sent = [(w + [0] * (self.max_word_length - len(w))) for w in sent]
        return sent

    def forward(self, x):
        # b: batch size
        # s: max words in a sentence
        # c: max characters in a word
        # o: output dimension
        # e: embed dimension

        batch_size = x.shape[0]

        # x : (b x s)

        # (b x s x c)
        x = torch.tensor([self.expand_sentence(s)
                                    for s in x])

        # (b x s x c x e)
        x = self.dropout(self.embedding(x))

        # (b x s x e x c)
        x = x.permute(0, 1, 3, 2)

        # (b*s x e x c)
        x = x.view(-1, self.embedding_dim, self.max_word_length)

        # (b*s x 1 x e x c)
        x = x.unsqueeze(1)
        
        # (b*s x o x 1 x W_out)
        x = self.relu(self.char_conv(x))

        # (b*s x o x W_out)
        x = x.squeeze()

        # (b*s x o)
        x = F.max_pool1d(x, x.shape[2]).squeeze()

        # (b x s x o)
        x = x.view(batch_size, -1, x.shape[-1])
        
        return x