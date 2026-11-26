class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        words = set()
        for t in texts:
            words.update(t.lower().split())
        tokens = specials + sorted(words)
        self.word_to_id = {token:i for i,token in enumerate(tokens)}
        self.id_to_word = {i:token for i,token in enumerate(tokens)}
        self.vocab_size = len(tokens)
        
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(t, unk_id) for t in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return ' '.join([self.id_to_word.get(id, self.unk_token) for id in ids])