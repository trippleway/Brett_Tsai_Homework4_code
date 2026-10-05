from tokenizers.input_text import INPUT_TEXT


chars = sorted(list(set(INPUT_TEXT)))
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}
encoded = [stoi[char] for char in INPUT_TEXT]
decoded = "".join(itos[index] for index in encoded)

print("Character-token count:", len(encoded))
print("Round-trip exact:", decoded == INPUT_TEXT)