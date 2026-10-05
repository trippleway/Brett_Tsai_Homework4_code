import string
import unicodedata

from tokenizers.input_text import INPUT_TEXT


cleaned = "".join(
    char
    for char in INPUT_TEXT.lower()
    if char not in string.punctuation
    and not unicodedata.category(char).startswith("P")
)
tokens = cleaned.split()
vocabulary = sorted(set(tokens))
word_to_id = {word: index for index, word in enumerate(vocabulary)}
id_to_word = {index: word for word, index in word_to_id.items()}
encoded = [word_to_id[word] for word in tokens]
decoded_tokens = [id_to_word[index] for index in encoded]
decoded_sentence = " ".join(decoded_tokens)

print("Tokens:", tokens)
print("Vocabulary size:", len(vocabulary))
print("Word-token count:", len(encoded))
print("Decoded sentence:", decoded_sentence)