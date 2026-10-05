import numpy as np
import tensorflow as tf


SEED = 1710
tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)

LINES = [
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
TEXT = "\n".join(LINES)
CHARS = sorted(set(TEXT))
CHAR_TO_ID = {char: index for index, char in enumerate(CHARS)}
ID_TO_CHAR = dict(enumerate(CHARS))
WINDOW = 30

inputs = []
targets = []
for start in range(len(TEXT) - WINDOW):
    inputs.append([CHAR_TO_ID[char] for char in TEXT[start : start + WINDOW]])
    targets.append(CHAR_TO_ID[TEXT[start + WINDOW]])

X = np.asarray(inputs, dtype=np.int32)
y = np.asarray(targets, dtype=np.int32)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(len(CHARS), 16),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(len(CHARS)),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
history = model.fit(X, y, batch_size=64, epochs=12, verbose=0)


def probabilities_for(prompt, temperature):
    padded = prompt.rjust(WINDOW)[-WINDOW:]
    context = [CHAR_TO_ID.get(char, 0) for char in padded]
    logits = model.predict(np.asarray([context], dtype=np.int32), verbose=0)[0]
    if temperature <= 0:
        result = np.zeros_like(logits, dtype=np.float64)
        result[np.argmax(logits)] = 1.0
        return result
    return tf.nn.softmax(logits / temperature).numpy()


def generate(prompt="I like ", count=120, temperature=0.7):
    padded = prompt.rjust(WINDOW)[-WINDOW:]
    context = [CHAR_TO_ID.get(char, 0) for char in padded]
    output = list(prompt)
    for _ in range(count):
        logits = model.predict(np.asarray([context], dtype=np.int32), verbose=0)[0]
        if temperature <= 0:
            next_id = int(np.argmax(logits))
        else:
            distribution = tf.nn.softmax(logits / temperature).numpy()
            next_id = int(rng.choice(len(distribution), p=distribution))
        output.append(ID_TO_CHAR[next_id])
        context = context[1:] + [next_id]
    return "".join(output)


print("Corpus length:", len(TEXT))
print("Vocabulary:", CHARS)
print("Training samples:", len(X))
print("Final loss:", history.history["loss"][-1])
for temperature in (0.1, 0.5, 0.7, 1.0):
    distribution = probabilities_for("I like ", temperature)
    top_ids = np.argsort(distribution)[-5:][::-1]
    print(f"\n=== Temperature {temperature} ===")
    print("Top next characters:")
    for index in top_ids:
        print(f"  {ID_TO_CHAR[int(index)]!r}: {distribution[index]:.1%}")
    print(generate(temperature=temperature))