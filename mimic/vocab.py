import json
from pathlib import Path
from typing import Self


class Vocab(object):
    """
    This class records all encountered tokens and how many times each has been
    encountered. Very helpful in reducing the size of processed data.
    """

    def __init__(self, sep_token: str = '[sep]'):
        """
        Constructor for vocab.

        Args:
            sep_token (str, optional): The token used to separate sequences. Defaults
                to '[sep]'.
        """

        self.tok2id: dict[str, int] = {}  # Map a token to its ID
        self.id2tok: dict[int, str] = {}  # Map a ID to its token
        self.tok2cnt: dict[str, int] = {}  # How many times a token has been added.
        self.cnt: int = 1  # Number of unique tokens.
        self.sep_token = sep_token  # Seperating token.

    def add_token(self, token: str):
        """
        Add a token. If the token hasn't been encountered before an ID will be assigned
        to it.

        Args:
            token (str): Token to add.
        """

        if token not in self.tok2id:
            self.tok2id[token] = self.cnt
            self.id2tok[self.cnt] = token
            self.tok2cnt[token] = 1
            self.cnt += 1
        else:
            self.tok2cnt[token] += 1

    def top_tokens(self, top: int) -> set[str]:
        """
        Grab the `top` highest count tokens recorded in this vocab.

        Args:
            top (int): Number of tokens to return/

        Returns:
            set[str]: A set of the `top` highest count tokens.
        """

        return set([tok
                    for _, tok in
                    sorted(list({v: k for k, v in self.tok2cnt.items()}.items()),
                           reverse=True)
                    [:top]])

    def to_json(self, path: str | Path):
        """
        Save the vocab to a JSON file.

        Args:
            path (str | Path): The path of the JSON file.
        """

        vocab_data = {
            'tok2id': self.tok2id,
            'id2tok': self.id2tok,
            'tok2cnt': self.tok2cnt,
            'cnt': self.cnt
        }

        with open(str(path), 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=4)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """
        Construct a Vocab object from a JSON file.

        Args:
            path (str | Path): Path to the JSON file.

        Returns:
            Vocab: A vocab object.
        """

        with open(str(path), 'r') as f:
            vocab_data = json.load(f)

        v = cls()
        v.tok2id = {k: int(v) for k, v in vocab_data['tok2id'].items()}
        v.id2tok = {int(k): v for k, v in vocab_data['id2tok'].items()}
        v.tok2cnt = {k: int(v) for k, v in vocab_data['tok2cnt'].items()}
        v.cnt = vocab_data['cnt']

        return v

    def __len__(self) -> int:
        """
        The number of unique tokens in the vocab.

        Returns:
            int: The number of unique tokens in the vocab.
        """

        return self.cnt
