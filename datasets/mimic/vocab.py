import json

SEP_TOKEN = '[SEP]'


class Vocab(object):
    def __init__(self):
        self.tok2id = {}
        self.id2tok = {}
        self.tok2cnt = {}
        self.cnt = 1

    def add_token(self, token: str):
        if token not in self.tok2id:
            self.tok2id[token] = self.cnt
            self.id2tok[self.cnt] = token
            self.tok2cnt[token] = 1
            self.cnt += 1
        else:
            self.tok2cnt[token] += 1

    def top_tokens(self, top: int):
        return set([tok for _, tok in sorted(list({v: k for k, v in self.tok2cnt.items()}.items()), reverse=True)[:top]])

    def to_json(self, path: str):
        vocab_data = {
            'tok2id': self.tok2id,
            'id2tok': self.id2tok,
            'tok2cnt': self.tok2cnt,
            'cnt': self.cnt
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=4)

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        v = Vocab()
        v.tok2id = {k: int(v) for k, v in vocab_data['tok2id'].items()}
        v.id2tok = {int(k): v for k, v in vocab_data['id2tok'].items()}
        v.tok2cnt = {k: int(v) for k, v in vocab_data['tok2cnt'].items()}
        v.cnt = vocab_data['cnt']

        return v

    def __len__(self):
        return self.cnt
