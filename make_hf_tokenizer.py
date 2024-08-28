"""
Usage
-----
python make_hf_tokenizer.py --output tinytokenizer.json  --vocab vocab/wordbank-en.txt vocab/ch-hf.txt


#Load it:

tokenizer = load_tokenizer('tinytokenizer.json')
print(" ".join(tokenizer.encode("They go to the beach").tokens))


#To just normalize+pre-tokenize (keeping any OOD words):

tokenize_keep_ood(tokenizer, text)

"""
import argparse
import re
import json
import argparse
from typing import List
from nltk.tokenize.util import align_tokens
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
    NormalizedString,
    PreTokenizedString
)
from tokenizers.pre_tokenizers import PreTokenizer

#NOTE: this function from https://github.com/noanabeshima/tiny_tokenizer/tree/main
single_character_toks = re.compile(r'[ ,—\*;:\."\'0-9?!\(\)\n-]')

def word_tokenize(text, known_toks=None, add_bos=True):
    """The function used for pre-tokenization."""
    matches = list(single_character_toks.finditer(text))

    i = 0
    res = []
    for match in matches:
        span = match.span()
        assert span[0] >= i
        assert span[1] > span[0]

        if span[0] > i:
            res.append(text[i:span[0]])
        res.append(text[span[0]:span[1]])
        i = span[1]
    if i < len(text):
        res.append(text[i:])

    res = [substr for substr in res if substr != ' ']
    if known_toks is not None:
        res = [tok if tok in known_toks else f'[unk]' for tok in res]

    if add_bos:
        res = ['[bos]'] + res
    return res


def get_tokens_and_offsets(text):
    tokens = word_tokenize(text, add_bos=False)
    token_positions = align_tokens(tokens, text)

    assert len(tokens) == len(token_positions)
    tokpos = [(tokens[ii], token_positions[ii][0], token_positions[ii][1]) for ii in range(len(tokens))]
    return tokpos


class CustomPreTokenizer:
    def _split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        tokpos = get_tokens_and_offsets(str(normalized_string))

        for token, start, stop in tokpos:
            splits.append(normalized_string[start:stop])

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self._split)


def load_vocab_files(filenames):
    L = []
    for f in filenames:
        with open(f,'r') as fd:
            v = fd.readlines()
            v = [i.strip().split('\t')[0] for i in v]
        L = L + v
    return list(set(L))


def make_tokenizer(vocab_files):
    """
    Make a HF tokenizer based on a vocab list given in the files `vocab_files`.

    Note:
    - WordLevel model used since we are not doing BPE.
    - Normalizer: lowercase, strip accents, normalize quotes and apostrophes.
    - Pretokenizer: uses function `word_tokenize`

    """
    vocab = load_vocab_files(vocab_files)
    vd = dict([(i,ii+1) for ii,i in enumerate(vocab)])
    vd['[UNK]'] = 0

    tokenizer = Tokenizer(models.WordLevel(vocab = vd, unk_token = "[UNK]" ) )

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace("“", '"'),
            normalizers.Replace("”", '"'),
            normalizers.Replace("’", "'"),
            normalizers.Replace("‘", "'"),
            normalizers.Replace("–", "—"),
            normalizers.NFKD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
    )

    tokenizer.pre_tokenizer = PreTokenizer.custom(CustomPreTokenizer())
    return tokenizer


def tokenize_keep_ood(tokenizer, text):
    """Just do normalization and pretokenization.
    Useful if we want to keep OOD and just see what the word splits are."""

    return [i[0] for i in tokenizer.pre_tokenizer.pre_tokenize_str(tokenizer.normalizer.normalize_str(text))]


def save_tokenizer(tokenizer, filename='tinytokenizer.json'):
    #note only way to save it since I used a custom pre-tokenizer:
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(filename)


def load_tokenizer(filename):
    tokenizer = Tokenizer.from_file(filename)
    tokenizer.pre_tokenizer = PreTokenizer.custom(CustomPreTokenizer())
    return tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a custom huggingface tokenizer from a given vocab.")

    # Add arguments
    parser.add_argument("--output", type=str, default='tinytokenizer.json', help="Filename in which to save the json for the tokenizer.")
    parser.add_argument("--vocab", nargs='+', type=str, help="List of vocab files; each word should be on a new line.")

    args = parser.parse_args()

    filename = args.output
    vocab_files = args.vocab

    tokenizer = make_tokenizer(vocab_files)
    save_tokenizer(tokenizer, filename=filename)
