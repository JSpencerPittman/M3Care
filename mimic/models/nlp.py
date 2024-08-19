from torch import nn
from torchtyping import TensorType
from torch.nn import functional as F
import torch
from typing import Sequence

from mimic.vocab import Vocab


BatchSequences = TensorType["batch_size", "seq_len"]


class WordEmbedder(nn.Module):
    def __init__(self,
                 vocab: Vocab,
                 vocab_size: int,
                 token_dim: int,
                 embed_dim: int,
                 sep_token: str = '[sep]'):
        """
        Constructor for WordEmbedder.

        Args:
            vocab (Vocab): Vocab object containing mapping of tokens to their
                frequencies and IDs.
            vocab_size (int): Size of the vocabulary being used by the embedder. This
                is to deal with the tradeoff of inclusivity and model size.
            token_dim (int): The dimensioniality each token is directly mapped to.
            embed_dim (int): The dimensionality of the model. After the embedding layer
                the token is mapped via a Linear layer to the models dimensions.
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
        self.embedder = nn.Embedding(vocab_size+3, token_dim)
        self.linear = nn.Linear(token_dim, embed_dim)

    def forward(self, raw_seqs: Sequence[Sequence[int]]) -> BatchSequences:
        """
        Apply word embedder to a raw sequence.

        Args:
            raw_seqs (Sequence[Sequence[int]]): A batch of sequences containing token
                IDs as assigned by the Vocab object.

        Returns:
            BatchSequences: The embedded sequence.
        """

        raw_seqs = torch.tensor(
            [[self._tokid_to_embedded_idx(tok_id)
              for tok_id in seq]
             for seq in raw_seqs],
            dtype=torch.int32).to(self.embedder.weight.device)

        embedded = self.embedder(raw_seqs)
        return F.relu(self.linear(embedded))

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
