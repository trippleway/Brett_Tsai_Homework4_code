import string
import unicodedata

from tokenizers.input_text import INPUT_TEXT


def tokenize(text):
    without_punctuation = "".join(
        character
        for character in text.lower()
        if character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
    )
    return without_punctuation.split()


def build_vocabulary(tokens):
    return sorted(set(tokens))


def encode(tokens, vocabulary):
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    return [word_to_id[token] for token in tokens]


def decode(indices, vocabulary):
    return [vocabulary[index] for index in indices]


tokens = tokenize(INPUT_TEXT)
vocabulary = build_vocabulary(tokens)
token_ids = encode(tokens, vocabulary)
decoded_tokens = decode(token_ids, vocabulary)
decoded_sentence = " ".join(decoded_tokens)

assert decoded_tokens == tokens
print("Tokens:", tokens)
print("Vocabulary size:", len(vocabulary))
print("Word-token count:", len(token_ids))
print("Round-trip exact after normalization:", decoded_sentence == " ".join(tokens))
print("Decoded sentence:", decoded_sentence)