import torch
from torch import nn, Tensor

from m3care import tensortypes as tt
from mimic.models.sequential import SequentialEmbedder
from mimic.vocab import Vocab


class NLTokenEmbedder(nn.Module):
    """
    Generate embeddings for tokens.
    """

    def __init__(self,
                 vocab: Vocab,
                 embed_dim: int,
                 vocab_size: int = 5000,
                 sep_token: str = '[sep]'):
        """
        Constructor for TokenEmbedder.

        Args:
            vocab (Vocab): Vocab object containing mapping of tokens to their
                frequencies and IDs.
            vocab_size (int): Size of the vocabulary being used by the embedder. This
                is to deal with the tradeoff of inclusivity and model size.
            embed_dim (int): The dimensioniality of the embedded token.
            sep_token (str, optional): The token used to signify a separation. Defaults
                to '[sep]'.
        """

        super().__init__()

        active_tokens = vocab.top_tokens(vocab_size)
        self.tokid_to_embed_idx = {
            vocab.tok2id[tok]: idx for idx, tok in enumerate(active_tokens, start=1)
        }
        self.sep_tok_id = vocab.tok2id[sep_token]
        self.vocab_size = vocab_size

        # Embedder Input Range:
        # 0       -> Separator token.
        # 1,...,V -> Vocabulary token.
        # V+1     -> Uknown token.
        # V+2     -> Padding token.
        self.embedder = nn.Embedding(vocab_size+3, embed_dim)

    def forward(self, raw_seqs: tt.BatSeqTensor) -> tt.BatSeqEmbTensor:
        """
        Apply word embedder to a raw sequence.

        Args:
            raw_seqs (BatSeqTensor): A batch of sequences containing token
                IDs as assigned by the Vocab object.

        Returns:
            BatSeqEmbTensor: The embedded sequence.
        """

        raw_seqs = torch.tensor(
            [[self._tokid_to_embedded_idx(tok_id)
              for tok_id in seq]
             for seq in raw_seqs],
            dtype=torch.int32).to(self.embedder.weight.device)

        return self.embedder(raw_seqs)

    def _tokid_to_embedded_idx(self, tok_id: int) -> int:
        """
        Each token must be mapped to the same embedded index prior to be passed through
        the embedder.

        Args:
            tok_id (int): The ID of the token (Vocab's assigned ID).

        Returns:
            int: The token's index for the embedder.
        """

        # Seperator token.
        if tok_id == self.sep_tok_id:
            return 0
        # Vocabulary token.
        elif tok_id in self.tokid_to_embed_idx:
            return self.tokid_to_embed_idx[tok_id]
        # Padding token.
        elif tok_id == 0:
            return self.vocab_size+2
        # Unknown token.
        else:
            return self.vocab_size+1


class NLSequenceEmbedder(nn.Module):
    def __init__(self,
                 vocab: Vocab,
                 token_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 vocab_size: int = 5000,
                 max_len: int = 5000,
                 sep_token: str = '[sep]',
                 dropout: float = 0.1):
        """
        Constructor for NLSequenceEmbedder.

        Args:
            vocab (Vocab): The vocab to translate tokens to IDs.
            token_dim (int): The dimension each token is represented with.
            embed_dim (int): The final dimensionality of the embedded sequence.
            num_heads (int): Number of heads in the transformer.
            vocab_size (int, optional): How many of the top words to use in the token
                embedder. Defaults to 5000.
            max_len (int, optional): Max length of a sequence. Defaults to 5000.
            sep_token (str, optional): The token used to signify a separation. Defaults
                to '[sep]'.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__()

        self.tok_embedder = NLTokenEmbedder(vocab,
                                            token_dim,
                                            vocab_size,
                                            sep_token)
        self.seq_emb = SequentialEmbedder(token_dim,
                                          embed_dim,
                                          dropout,
                                          max_len,
                                          num_heads)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, tokens: tt.BatSeqTensor, mask: tt.BatSeqTensor
                ) -> tuple[tt.BatEmbTensor, tt.BatTensor]:
        tok_emb = self.tok_embedder(tokens)
        emb, _ = self.seq_emb(tok_emb, mask)
        emb = self.avg_pool(emb.transpose(1, 2)).squeeze()
        mask = mask.int().sum(axis=-1) > 0

        return emb, mask


class NLTimeSeriesSequenceEmbedder(nn.Module):
    def __init__(self,
                 vocab: Vocab,
                 token_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 vocab_size: int = 5000,
                 max_len: int = 5000,
                 sep_token: str = '[sep]',
                 dropout: float = 0.1):
        super().__init__()

        self.seq_emb = NLSequenceEmbedder(vocab,
                                          token_dim,
                                          embed_dim,
                                          num_heads,
                                          vocab_size,
                                          max_len,
                                          sep_token,
                                          dropout)

    def forward(self, tokens: tt.BatTimeSeqTensor, mask: tt.BatTimeSeqTensor
                ) -> tuple[tt.BatTimeEmbTensor, tt.BatTimeTensor]:
        batch_size, time_steps = tokens.shape[0:2]
        tokens, mask = tokens.flatten(0, 1), mask.flatten(0, 1)

        tokens, mask, existing_entries = self._compress(tokens, mask)
        emb, mask = self.seq_emb(tokens, mask)
        emb, mask = self._decompress(emb, mask, existing_entries)

        emb = emb.unflatten(0, (batch_size, time_steps))
        mask = mask.unflatten(0, (batch_size, time_steps))
        return emb, mask

    @staticmethod
    def _compress(tokens: tt.BatSeqTensor, mask: tt.BatSeqTensor
                  ) -> tuple[tt.BatSeqTensor, tt.BatSeqTensor, tt.BatTensor]:
        existing_entries = (mask.int().sum(-1) > 0)

        return tokens[existing_entries], mask[existing_entries], existing_entries

    @staticmethod
    def _decompress(emb: tt.BatEmbTensor,
                    mask: tt.BatTensor,
                    existing_entries: tt.BatTensor
                    ) -> tuple[tt.BatEmbTensor, tt.BatTensor]:
        # Allocate tensors
        num_entries, embed_dim = existing_entries.shape[0], emb.shape[-1]
        emb_dec = torch.zeros(num_entries, embed_dim)
        mask_dec = torch.BoolTensor(num_entries)

        # Copy contents
        existing_idxs = torch.tensor(
            [idx for idx, exists in enumerate(existing_entries) if exists]
            )
        for comp_idx, dec_idx in enumerate(existing_idxs):
            emb_dec[dec_idx], mask_dec[dec_idx] = emb[comp_idx], mask[comp_idx]

        return emb_dec, mask_dec
