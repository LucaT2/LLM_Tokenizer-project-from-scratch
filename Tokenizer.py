import regex as re
import unicodedata

from Hyperparameters import VOCAB_SIZE, NUM_MERGES, GPT4_SPLIT_PATTERN

# We will also define two functions such that we can better visualize our merges and vocab dictionaries
def replace_control_chars(s: str):
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_chars(s)
    return s


class Tokenizer:
    def __init__(self):
        self.regex_pattern = GPT4_SPLIT_PATTERN
        self.vocab_size = VOCAB_SIZE
        self.num_merges = NUM_MERGES
        self.merges = {}
        self.vocab = self.build_vocab()
        self.model_file = r"files\tokenizer.model"
        self.vocab_file = r"files\tokenizer.vocab"


    @staticmethod
    def get_stats(ids, counts = None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        ids = []
        text_split = re.findall(self.regex_pattern, text)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for chars in text_split:
            byte_list = list(chars.encode("utf-8"))
            ids.append(byte_list)

        for i in range(self.num_merges):
            stats = {}
            for chars in ids:
                self.get_stats(chars, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self.merge(chars, pair, idx) for chars in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda s: self.merges.get(s, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self,ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")  # replace inserts a special character for undecodable byte
        return text

    # This function builds the vocabulary from the merges that we perform
    def build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self):
        with open(self.model_file, 'w') as f:
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        inverted_merges = {idx: pair for pair, idx in
                           self.merges.items()}  # this is needed to find the children of each token that was newly created
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                # find the children of this token
                if idx in inverted_merges:
                    # if this token has children show the merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")  # this case is for the first 256 tokens

    def load(self):
        idx = 256
        with open(self.model_file, 'r', encoding="utf-8") as f:
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                self.merges[(idx1, idx2)] = idx
                idx += 1
        self.vocab = self.build_vocab()
