import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from m3care.cs224n.cnn import CNN
from m3care.cs224n.highway import Highway

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        char_embed_size = 50
        self.embed_size = embed_size
        self.vocab = vocab
        self.embeddings = nn.Embedding(len(vocab.char2id), char_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        self.cnn = CNN(char_embed_size, embed_size)
        self.highway = Highway(embed_size)


    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """

        data = self.embeddings(input) # (batch_size, sentence_length, max_word_length, char_embed_size)
        data_shape = data.shape
        data = data.reshape(data_shape[0] * data_shape[1], data_shape[2], data_shape[3]) # (batch_size * sentence_length, max_word_length, char_embed_size)
        x_word_embed = self.cnn(data.transpose(1, 2)) # x_word_embed is (batch_size * sentence_length, embed_size)
        x_word_embed = self.highway(x_word_embed)
        x_word_embed = self.dropout(x_word_embed)
        x_word_embed = x_word_embed.reshape(data_shape[0], data_shape[1], self.embed_size)
        return x_word_embed