# Tiny RNN text generator

import numpy as np
import tensorflow as tf


# Seed both TensorFlow and NumPy so initialization and character sampling can
# be reproduced. Use None when experimenting with different random outcomes.
# Keep the first run reproducible so classmates can compare results.
# Change this number—or set it to None—to experiment with different runs.
SEED = 1710
if SEED is not None:
    tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------- 1) Tiny corpus (swap this block to change tasks) ----------
# Each line is one short example. Joining with newlines keeps the boundary
# between examples visible to the character-level model.
tiny_lines = [
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
text = "\n".join(tiny_lines)

# Alternative corpora (replace the text above with one option to experiment):
# DNA: text = "TATAAA\nCGCGCG\nATG...TAA\nACGTACGTACGT\n"
# Emoji: text = "☀️🌤️⛅🌧️⛈️🌈\n🍞🧈🍯\n🥚🍳🍞\n🙂➡️😊\n"
# Nursery: text = "Twinkle twinkle little star,\nHow I wonder what you are.\n"

# The vocabulary contains each distinct character once. The two dictionaries
# let the model use integer IDs internally and turn predicted IDs back to text.
print("Corpus length:", len(text))
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)
print("Vocab:", chars)

# ---------- 2) Vectorize to (input sequence -> next char) pairs ----------
# Slide a fixed-size window through the corpus. Each input window is paired
# with the character immediately after it, which is the prediction target.
seq_len = 40
step = 1
X_idx, y_idx = [], []
for i in range(0, len(text) - seq_len, step):
    seq = text[i : i + seq_len]
    nxt = text[i + seq_len]
    X_idx.append([stoi[c] for c in seq])
    y_idx.append(stoi[nxt])

# Integer arrays are the format expected by the embedding layer and loss.
X = np.array(X_idx, dtype=np.int32)
y = np.array(y_idx, dtype=np.int32)
print("Num training samples:", len(X))

# ---------- 3) Model ----------
# Embedding turns each character ID into a learned vector; the LSTM reads the
# sequence and keeps its context; Dense produces one score (logit) per character.
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, 32),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(vocab_size),
    ]
)
# The labels are integer IDs, so use sparse cross-entropy. The model returns
# raw logits, which is why from_logits is enabled instead of adding softmax here.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# ---------- 4) Train (tiny & fast) ----------
# fit adjusts the weights to predict the next character for every training pair.
history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
print("Final loss:", history.history["loss"][-1])


# ---------- 5) Sampling helper with temperature ----------
def sample_logits(logits, temperature=1.0):
    # A non-positive temperature selects the highest-scoring character.
    if temperature <= 0:  # greedy
        return int(np.argmax(logits))
    # Softmax converts scaled logits to probabilities. Smaller temperatures
    # concentrate probability on likely choices; larger ones spread it out.
    logits = logits / temperature
    probabilities = tf.nn.softmax(logits).numpy()
    return int(rng.choice(len(probabilities), p=probabilities))


def next_char_probabilities(seed="I like ", temperature=1.0, top_n=5):
    """Return the most likely next characters for a prompt."""
    # Pad short prompts on the left so prediction always receives seq_len IDs.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed)
    # Unknown prompt characters fall back to ID 0; the model predicts logits
    # for the next character after the final context window.
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    x = np.array([context], dtype=np.int32)
    logits = model.predict(x, verbose=0)[0]

    if temperature <= 0:
        probabilities = np.zeros_like(logits, dtype=np.float64)
        probabilities[np.argmax(logits)] = 1.0
    else:
        probabilities = tf.nn.softmax(logits / temperature).numpy()

    # Sort by probability, highest first, then map IDs back to characters.
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    return [(itos[int(i)], float(probabilities[i])) for i in top_indices]


def generate(seed="I like ", n_chars=200, temperature=0.7):
    # Ensure seed length is at least seq_len by left-padding with spaces.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed)
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    output = list(seed)
    for _ in range(n_chars):
        # Predict one character, append it, and slide the context forward.
        # Feeding each prediction back makes generation autoregressive.
        x = np.array([context], dtype=np.int32)
        logits = model.predict(x, verbose=0)[0]
        idx = sample_logits(logits, temperature)
        char = itos[idx]
        output.append(char)
        context = context[1:] + [idx]
    return "".join(output)


# ---------- 6) Try a few temperatures ----------
# Compare how the same trained model samples at different levels of randomness.
for temperature in [0.1, 0.5, 0.7, 1.0]:
    print("\n=== Temperature", temperature, "===")
    print("Top probabilities for the first generated character:")
    for char, probability in next_char_probabilities(
        seed="I like ", temperature=temperature
    ):
        print(f"  {char!r}: {probability:.1%}")
    print(generate(seed="I like ", n_chars=180, temperature=temperature))
