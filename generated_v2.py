import numpy as np
import tensorflow as tf


SEED = 1710
tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)

TINY_LINES = [
    "I like cats.",
    "I like dogs.",
    "I like noodles.",
    "I like tacos.",
    "I like books.",
    "I like robots.",
    "I like music.",
    "I like puzzles.",
    "I like pizza.",
    "I like coding.",
]
TEXT = "\n".join(TINY_LINES)
CHARACTERS = sorted(set(TEXT))
CHAR_TO_INDEX = {char: index for index, char in enumerate(CHARACTERS)}
INDEX_TO_CHAR = {index: char for char, index in CHAR_TO_INDEX.items()}
VOCAB_SIZE = len(CHARACTERS)
CONTEXT_LENGTH = 40

training_inputs = []
training_targets = []
for start in range(len(TEXT) - CONTEXT_LENGTH):
    context = TEXT[start : start + CONTEXT_LENGTH]
    next_character = TEXT[start + CONTEXT_LENGTH]
    training_inputs.append([CHAR_TO_INDEX[char] for char in context])
    training_targets.append(CHAR_TO_INDEX[next_character])

X = np.asarray(training_inputs, dtype=np.int32)
y = np.asarray(training_targets, dtype=np.int32)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(VOCAB_SIZE, 32),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(VOCAB_SIZE),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
training_history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)


def context_indices(prompt):
    padded_prompt = prompt if len(prompt) >= CONTEXT_LENGTH else (
        " " * (CONTEXT_LENGTH - len(prompt)) + prompt
    )
    return [CHAR_TO_INDEX.get(char, 0) for char in padded_prompt[-CONTEXT_LENGTH:]]


def next_character_probabilities(prompt="I like ", temperature=1.0):
    context = context_indices(prompt)
    logits = model.predict(np.asarray([context], dtype=np.int32), verbose=0)[0]
    if temperature <= 0:
        probabilities = np.zeros_like(logits, dtype=np.float64)
        probabilities[np.argmax(logits)] = 1.0
        return probabilities
    return tf.nn.softmax(logits / temperature).numpy()


def generate(prompt="I like ", character_count=180, temperature=0.7):
    context = context_indices(prompt)
    output = list(prompt)
    for _ in range(character_count):
        logits = model.predict(np.asarray([context], dtype=np.int32), verbose=0)[0]
        if temperature <= 0:
            next_index = int(np.argmax(logits))
        else:
            probabilities = tf.nn.softmax(logits / temperature).numpy()
            next_index = int(rng.choice(len(probabilities), p=probabilities))
        output.append(INDEX_TO_CHAR[next_index])
        context = context[1:] + [next_index]
    return "".join(output)


print("Corpus length:", len(TEXT))
print("Vocab:", CHARACTERS)
print("Num training samples:", len(X))
print("Final loss:", training_history.history["loss"][-1])

for temperature in (0.1, 0.5, 0.7, 1.0):
    probabilities = next_character_probabilities("I like ", temperature)
    top_indices = np.argsort(probabilities)[-5:][::-1]
    print(f"\n=== Temperature {temperature} ===")
    print("Top probabilities for the first generated character:")
    for index in top_indices:
        print(f"  {INDEX_TO_CHAR[int(index)]!r}: {probabilities[index]:.1%}")
    print(generate("I like ", character_count=180, temperature=temperature))