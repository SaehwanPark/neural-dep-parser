# `config.py` (python)

```python
from typing import NamedTuple
from schema import ParserVocab


class ParserConfig(NamedTuple):
  """Constants and special IDs for the parser logic."""

  n_features: int = 36
  hidden_size: int = 200
  n_classes: int = 3
  embed_size: int = 50
  dropout_rate: float = 0.5

  # Special IDs mapped from ParserVocab
  NULL_ID: int = 0
  P_NULL_ID: int = 0
  ROOT_ID: int = 0
  P_ROOT_ID: int = 0
  UNK_ID: int = 0
  P_UNK_ID: int = 0


def create_config(vocab: ParserVocab) -> ParserConfig:
  """Factory function to populate IDs based on the actual vocab."""
  return ParserConfig(
    NULL_ID=vocab.word2id["<NULL>"],
    P_NULL_ID=vocab.pos2id["<p>:<NULL>"],
    ROOT_ID=vocab.word2id["<ROOT>"],
    P_ROOT_ID=vocab.pos2id["<p>:<ROOT>"],
    UNK_ID=vocab.word2id["<UNK>"],
    P_UNK_ID=vocab.pos2id["<p>:<UNK>"],
  )

```


# `data_loader.py` (python)

```python
import os
import numpy as np
import jax.numpy as jnp
from dotenv import load_dotenv
from schema import Sentence, ParserVocab
from typing import List, Dict

load_dotenv()


def load_conll_data(file_name: str, lowercase: bool = True) -> List[Dict]:
  """
  Reads CoNLL file into a list of raw dictionaries.
  Pure function: no side effects other than file reading.
  """
  data_path = os.getenv("DATA_PATH", "./data")
  full_path = os.path.join(data_path, file_name)

  examples = []
  with open(full_path, "r", encoding="utf-8") as f:
    word, pos, head, label = [], [], [], []
    for line in f:
      sp = line.strip().split("\t")
      if len(sp) == 10:
        if "-" not in sp[0]:  # Skip multi-word tokens
          word.append(sp[1].lower() if lowercase else sp[1])
          pos.append(sp[4])
          head.append(int(sp[6]))
          label.append(sp[7])
      elif len(word) > 0:
        examples.append({"word": word, "pos": pos, "head": head, "label": label})
        word, pos, head, label = [], [], [], []
  return examples


def build_vocab(train_data: List[Dict]) -> ParserVocab:
  """Builds vocabularies with unique ID offsets for each feature type."""
  # 1. Labels (L_PREFIX)
  # root_labels = [
  #   ex["label"][i] for ex in train_data for i, h in enumerate(ex["head"]) if h == 0
  # ]
  # root_label = Counter(root_labels).most_common(1)[0][0]

  unique_labels = list(set([ll for ex in train_data for ll in ex["label"]]))
  label2id = {f"<l>:{ll}": i for i, ll in enumerate(unique_labels)}
  label2id["<l>:<NULL>"] = len(label2id)
  id2label = {i: ll for ll, i in label2id.items()}

  # 2. POS Tags (P_PREFIX) with offset
  pos_offset = len(label2id)
  all_pos = [f"<p>:{p}" for ex in train_data for p in ex["pos"]]
  unique_pos = list(set(all_pos))
  pos2id = {p: i + pos_offset for i, p in enumerate(unique_pos)}
  pos2id["<p>:<UNK>"] = len(pos2id) + pos_offset
  pos2id["<p>:<NULL>"] = len(pos2id) + pos_offset + 1
  pos2id["<p>:<ROOT>"] = len(pos2id) + pos_offset + 2

  # 3. Words with offset
  word_offset = pos_offset + len(pos2id)
  all_words = [w for ex in train_data for w in ex["word"]]
  unique_words = list(set(all_words))
  word2id = {w: i + word_offset for i, w in enumerate(unique_words)}
  word2id["<UNK>"] = len(word2id) + word_offset
  word2id["<NULL>"] = len(word2id) + word_offset + 1
  word2id["<ROOT>"] = len(word2id) + word_offset + 2

  return ParserVocab(word2id, pos2id, label2id, id2label)


def vectorize_sentences(
  raw_data: List[Dict], vocab: ParserVocab, max_len: int = 120
) -> List[Sentence]:
  """
  Converts raw string data into fixed-width jnp arrays.
  Corrected to handle specific prefix keys and padding correctly.
  """
  sentences = []
  for ex in raw_data:
    n = len(ex["word"])
    # Padding logic: Using -1 for position padding to distinguish from index 0 (ROOT)
    words = np.zeros(max_len, dtype=np.int32)
    pos = np.zeros(max_len, dtype=np.int32)
    heads = np.full(max_len, -1, dtype=np.int32)

    # 1. Map tokens to IDs using the specific prefixes defined in build_vocab
    for i in range(min(n, max_len)):
      # Word mapping
      words[i] = vocab.word2id.get(ex["word"][i], vocab.word2id["<UNK>"])

      # POS mapping - corrected to use the "<p>:" prefix defined in build_vocab
      pos_key = f"<p>:{ex['pos'][i]}"
      pos[i] = vocab.pos2id.get(pos_key, vocab.pos2id["<p>:<UNK>"])

      # Head mapping
      heads[i] = ex["head"][i]

    # 2. Create boolean mask for JAX-compatible loss calculation
    mask = np.arange(max_len) < n

    sentences.append(
      Sentence(
        words=jnp.array(words),
        pos=jnp.array(pos),
        heads=jnp.array(heads),
        # Labels are currently simplified to zeros as per previous implementation
        labels=jnp.zeros(max_len, dtype=np.int32),
        mask=jnp.array(mask),
      )
    )
  return sentences

```


# `engine.py` (python)

```python
import jax
import jax.numpy as jnp
from schema import ParserState, Sentence
from config import ParserConfig


def init_state(sentence_len: int, max_stack: int, max_buffer: int) -> ParserState:
  """
  Initializes the ParserState for a new sentence.
  """
  # The buffer initially contains all word indices (1 to sentence_len)
  # The stack starts with the ROOT token (index 0)
  return ParserState(
    stack=jnp.zeros(max_stack, dtype=jnp.int32).at[0].set(0),
    buffer=jnp.arange(max_buffer, dtype=jnp.int32),
    dependencies=jnp.full((max_buffer, 2), -1, dtype=jnp.int32),
    stack_ptr=0,  # Points to the top of the stack (0-based)
    buffer_ptr=1,  # Points to the next available word in the buffer
    num_deps=0,
  )


@jax.jit
def apply_transition(state: ParserState, transition_id: int) -> ParserState:
  """
  A pure function that transforms the current state based on an action.
  0: Shift (S), 1: Left-Arc (LA), 2: Right-Arc (RA)
  """

  def shift_fn():
    # Push buffer[buffer_ptr] onto stack, increment pointers
    next_word = state.buffer[state.buffer_ptr]
    new_stack = state.stack.at[state.stack_ptr + 1].set(next_word)
    return state._replace(
      stack=new_stack, stack_ptr=state.stack_ptr + 1, buffer_ptr=state.buffer_ptr + 1
    )

  def left_arc_fn():
    # dependent = stack[top-1], head = stack[top]
    # Remove stack[top-1] by shifting the top element down
    head = state.stack[state.stack_ptr]
    dep = state.stack[state.stack_ptr - 1]
    new_deps = state.dependencies.at[state.num_deps].set(jnp.array([head, dep]))
    new_stack = state.stack.at[state.stack_ptr - 1].set(head)
    return state._replace(
      stack=new_stack,
      stack_ptr=state.stack_ptr - 1,
      dependencies=new_deps,
      num_deps=state.num_deps + 1,
    )

  def right_arc_fn():
    # head = stack[top-1], dependent = stack[top]
    # Remove stack[top]
    head = state.stack[state.stack_ptr - 1]
    dep = state.stack[state.stack_ptr]
    new_deps = state.dependencies.at[state.num_deps].set(jnp.array([head, dep]))
    return state._replace(
      stack_ptr=state.stack_ptr - 1, dependencies=new_deps, num_deps=state.num_deps + 1
    )

  return jax.lax.switch(transition_id, [shift_fn, left_arc_fn, right_arc_fn])


@jax.jit
def extract_features(
  state: ParserState, sentence: Sentence, config: ParserConfig
) -> jnp.ndarray:
  """
  Extracts 36 features (18 word IDs, 18 POS IDs) from the current parser state.
  """

  # Helper to find children in the current dependency list
  def get_child(head_idx, side, rank):
    # side: 0 for left, 1 for right; rank: 0 for 1st, 1 for 2nd
    # dependencies shape: (max_deps, 2) -> [head, dependent]
    mask = state.dependencies[:, 0] == head_idx
    # Valid dependents are those matching the head and on the correct side
    if side == 0:  # Left: dependent index < head index
      match_mask = (
        mask & (state.dependencies[:, 1] < head_idx) & (state.dependencies[:, 1] != -1)
      )
      # Sort ascending to find the leftmost (smallest index)
      matches = jnp.sort(jnp.where(match_mask, state.dependencies[:, 1], 999))
    else:  # Right: dependent index > head index
      match_mask = mask & (state.dependencies[:, 1] > head_idx)
      # Sort descending to find the rightmost (largest index)
      matches = jnp.sort(jnp.where(match_mask, state.dependencies[:, 1], -1))[::-1]

    child_idx = matches[rank]
    return jnp.where((child_idx == 999) | (child_idx == -1), -1, child_idx)

  # 1. Basic Stack and Buffer positions using jnp.where
  s0 = state.stack[state.stack_ptr]

  # Use jnp.where instead of Python if/else
  s1 = jnp.where(
    state.stack_ptr >= 1, state.stack[jnp.maximum(0, state.stack_ptr - 1)], -1
  )
  s2 = jnp.where(
    state.stack_ptr >= 2, state.stack[jnp.maximum(0, state.stack_ptr - 2)], -1
  )

  b0 = jnp.where(
    state.buffer_ptr < state.buffer.shape[0], state.buffer[state.buffer_ptr], -1
  )
  b1 = jnp.where(
    state.buffer_ptr + 1 < state.buffer.shape[0],
    state.buffer[jnp.minimum(state.buffer.shape[0] - 1, state.buffer_ptr + 1)],
    -1,
  )
  b2 = jnp.where(
    state.buffer_ptr + 2 < state.buffer.shape[0],
    state.buffer[jnp.minimum(state.buffer.shape[0] - 1, state.buffer_ptr + 2)],
    -1,
  )

  # 2. First and second order children logic
  lc_s0, rc_s0 = get_child(s0, 0, 0), get_child(s0, 1, 0)
  lc_s1, rc_s1 = get_child(s1, 0, 0), get_child(s1, 1, 0)

  # Second leftmost/rightmost children
  lc2_s0, rc2_s0 = get_child(s0, 0, 1), get_child(s0, 1, 1)
  lc2_s1, rc2_s1 = get_child(s1, 0, 1), get_child(s1, 1, 1)

  # Grandchildren (leftmost of leftmost, rightmost of rightmost)
  llc_s0, rrc_s0 = get_child(lc_s0, 0, 0), get_child(rc_s0, 1, 0)
  llc_s1, rrc_s1 = get_child(lc_s1, 0, 0), get_child(rc_s1, 1, 0)

  # 3. Consolidate 18 positions
  pos_indices = jnp.array(
    [
      s0,
      s1,
      s2,
      b0,
      b1,
      b2,
      lc_s0,
      rc_s0,
      lc2_s0,
      rc2_s0,
      llc_s0,
      rrc_s0,
      lc_s1,
      rc_s1,
      lc2_s1,
      rc2_s1,
      llc_s1,
      rrc_s1,
    ]
  )

  # 4. Map positions to IDs using Sentence data and Vocab NULLs
  # If index is -1 (meaning the stack/buffer/child doesn't exist), use NULL IDs
  word_features = jnp.where(
    pos_indices >= 0, sentence.words[pos_indices], config.NULL_ID
  )
  pos_features = jnp.where(
    pos_indices >= 0, sentence.pos[pos_indices], config.P_NULL_ID
  )

  return jnp.concatenate([word_features, pos_features])


def get_legal_mask(state: ParserState) -> jnp.ndarray:
  """
  Returns a mask for legal transitions: [LA, RA, S]
  - Left-Arc: Legal if stack has at least 2 items (not including ROOT only)
  - Right-Arc: Legal if stack has at least 2 items
  - Shift: Legal if buffer is not empty
  """
  can_la = state.stack_ptr >= 2
  can_ra = state.stack_ptr >= 1
  can_shift = state.buffer_ptr < len(state.buffer)

  return jnp.array([can_la, can_ra, can_shift], dtype=jnp.float32)


def predict_action(logits: jnp.ndarray, state: ParserState) -> int:
  """
  Greedy selection of the best legal action.
  Uses a large negative value to mask illegal actions.
  """
  mask = get_legal_mask(state)
  # Large negative value for illegal moves
  masked_logits = logits + (1.0 - mask) * -1e9
  return jnp.argmax(masked_logits)

```


# `inference.py` (python)

```python
import jax
import jax.numpy as jnp
from engine import extract_features, apply_transition, predict_action, init_state


def minibatch_parse_step(params, states, sentences, model, config):
  """
  Performs one parsing step for a batch of sentences.
  Uses jax.vmap to parallelize across the batch dimension.
  """

  # 1. Vectorized feature extraction
  def get_feats(state, sent):
    return extract_features(state, sent, config)

  batch_feats = jax.vmap(get_feats)(states, sentences)

  # 2. Vectorized model prediction
  logits = model.apply({"params": params}, batch_feats, train=False)

  # 3. Vectorized action selection and state update
  def update_single_state(ll, s):
    action = predict_action(ll, s)
    return apply_transition(s, action)

  new_states = jax.vmap(update_single_state)(logits, states)
  return new_states


def is_finished(states):
  """Checks if all sentences in the batch have empty buffers."""
  return jnp.all(states.buffer_ptr >= states.buffer.shape[1])


def calculate_uas(params, dev_sentences, model, config, batch_size=1024):
  total_correct = 0
  total_tokens = 0

  for i in range(0, len(dev_sentences), batch_size):
    batch_list = dev_sentences[i : i + batch_size]

    # 1. Properly stack the Sentences into a single Pytree of arrays
    # This ensures that batch.words has shape (batch_size, max_len)
    batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *batch_list)

    # 2. Initialize states and stack them
    # Ensure max_stack (30) and max_buffer (120) match your data_loader limits
    states_list = [init_state(len(s.words), 30, 120) for s in batch_list]
    states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states_list)

    # Iterative parsing loop
    steps = 0
    while not is_finished(states) and steps < 250:  # Added safety timeout
      states = minibatch_parse_step(params, states, batch, model, config)
      steps += 1

    # 3. Compare dependencies
    for b_idx in range(len(batch_list)):
      sent = batch_list[b_idx]
      predicted_heads = jnp.full(len(sent.words), -1)
      deps = states.dependencies[b_idx]

      for head, dep in deps:
        # Note: dep is the index of the child, head is the index of the parent
        # We use jnp.where or a check to avoid -1 indices
        if dep != -1:
          predicted_heads = predicted_heads.at[dep].set(head)

      # Calculate correct attachments, ignoring padding and ROOT (index 0)
      gold_heads = sent.heads
      mask = sent.mask & (jnp.arange(len(sent.words)) > 0)

      correct = jnp.sum((predicted_heads == gold_heads) & mask)
      total_correct += correct
      total_tokens += jnp.sum(mask)

  return float(total_correct) / float(total_tokens)

```


# `oracle.py` (python)

```python
import jax.numpy as jnp
from schema import ParserState, Sentence
from engine import extract_features, apply_transition


def get_gold_transition(state: ParserState, sentence: Sentence) -> int:
  # Logic: 0: Shift, 1: Left-Arc, 2: Right-Arc

  if state.stack_ptr < 1:
    return 0  # SHIFT (0)

  s1 = state.stack[state.stack_ptr]
  s2 = state.stack[state.stack_ptr - 1]

  if s2 > 0 and sentence.heads[s2] == s1:
    return 1  # LEFT-ARC (1)

  if sentence.heads[s1] == s2:
    has_buffer_dependents = jnp.any(sentence.heads[state.buffer_ptr :] == s1)
    if not has_buffer_dependents:
      return 2  # RIGHT-ARC (2)

  return 0  # SHIFT (0)


def oracle_step(state: ParserState, sentence: Sentence, config) -> tuple:
  """
  A single step of the oracle used for generating training instances.
  """
  features = extract_features(state, sentence, config)
  gold_action = get_gold_transition(state, sentence)
  next_state = apply_transition(state, gold_action)
  return features, gold_action, next_state

```


# `parser_model.py` (python)

```python
import flax.linen as nn


class ParserModel(nn.Module):
  """
  Flax implementation of the Feed-Forward Dependency Parser.
  """

  vocab_size: int
  embed_size: int = 50
  hidden_size: int = 200
  n_classes: int = 3
  dropout_rate: float = 0.5

  @nn.compact
  def __call__(self, x, train: bool = True):
    """
    x: (batch_size, n_features) - indices of features
    """
    # 1. Custom Embedding Lookup (Manual implementation as per requirement)
    # We use a learnable parameter for embeddings
    embeddings = self.param(
      "embeddings",
      nn.initializers.uniform(scale=0.1),
      (self.vocab_size, self.embed_size),
    )

    # Select embeddings and flatten: (batch, n_features * embed_size)
    # Equivalent to PyTorch's x.view(batch_size, -1)
    x = embeddings[x].reshape((x.shape[0], -1))

    # 2. Hidden Layer: Affine -> ReLU -> Dropout
    # We use Xavier Uniform (Glorot) for weights
    x = nn.Dense(
      features=self.hidden_size,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.uniform(),
    )(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

    # 3. Output Layer (Logits)
    logits = nn.Dense(
      features=self.n_classes,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.uniform(),
    )(x)

    return logits

```


# `run.py` (python)

```python
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dotenv import load_dotenv
from flax.training import train_state

from config import create_config
from data_loader import load_conll_data, build_vocab, vectorize_sentences
from parser_model import ParserModel
from engine import init_state
from oracle import oracle_step
from inference import calculate_uas
from utils import save_params, load_params


def create_learning_pipeline(model, params, learning_rate):
  """Initializes the Optax optimizer and Flax train state."""
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def loss_fn(params, batch_x, batch_y, model_apply_fn, dropout_rng):
  """
  Pure loss function.
  batch_x: (batch_size, n_features)
  batch_y: (batch_size,) - Gold transition indices
  """
  logits = model_apply_fn(
    {"params": params}, batch_x, train=True, rngs={"dropout": dropout_rng}
  )
  # Cross entropy loss between logits and gold transition labels
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y)
  return jnp.mean(loss)


@jax.jit
def train_step(state, batch_x, batch_y, dropout_rng):
  """Updates model parameters via gradient descent."""
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params, batch_x, batch_y, state.apply_fn, dropout_rng)
  state = state.apply_gradients(grads=grads)
  return state, loss


def init_train_state(model, rng, vocab_size, embed_size, learning_rate):
  """Initializes the Flax TrainState with parameters and optimizer."""
  variables = model.init(rng, jnp.ones((1, 36), dtype=jnp.int32), train=False)
  params = variables["params"]
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_minibatches(X, y, batch_size, shuffle=True):
  """
  Yields slices of data as minibatches.
  Purely functional approach to data iteration.
  """
  data_size = len(X)
  indices = np.arange(data_size)
  if shuffle:
    np.random.shuffle(indices)

  for start_idx in range(0, data_size, batch_size):
    end_idx = min(start_idx + batch_size, data_size)
    batch_indices = indices[start_idx:end_idx]
    yield X[batch_indices], y[batch_indices]


def main():
  # 1. Environment and Configuration
  load_dotenv()
  data_path = os.getenv("DATA_PATH", "/home/saehwan/data/dep_parser")

  # config = ParserConfig(
  #   n_features=36,
  #   hidden_size=200,
  #   n_classes=3,
  #   embed_size=50,
  #   dropout_rate=0.5
  # )

  # 2. Data Loading (Phase 1)
  print(f"Loading data from {data_path}...")
  raw_train = load_conll_data("train.conll")
  vocab = build_vocab(raw_train)
  config = create_config(vocab)

  # vectorize training data
  train_sentences = vectorize_sentences(raw_train, vocab)

  # load and vectorize the development + test sets
  raw_dev = load_conll_data("dev.conll")
  dev_sentences = vectorize_sentences(raw_dev, vocab)
  raw_test = load_conll_data("test.conll")
  test_sentences = vectorize_sentences(raw_test, vocab)

  # 3. Model Initialization (Phase 3)
  rng = jax.random.PRNGKey(0)
  model_rng, dropout_rng = jax.random.split(rng)
  max_id = max(max(vocab.word2id.values()), max(vocab.pos2id.values()))

  model = ParserModel(
    vocab_size=max_id + 1,
    embed_size=config.embed_size,
    hidden_size=config.hidden_size,
    n_classes=config.n_classes,
    dropout_rate=config.dropout_rate,
  )

  state = init_train_state(
    model, model_rng, model.vocab_size, config.embed_size, 0.0005
  )

  # 4. Generate Training Instances (Oracle logic)
  print("Generating training instances via Oracle...")
  # In a production setting, this would be lazily generated or cached
  # but here we follow the HW logic of pre-creating instances
  all_features = []
  all_labels = []

  for sent in train_sentences[:500]:  # Using a subset for demonstration
    p_state = init_state(len(sent.words), 30, 120)
    # Iterate until buffer is empty
    step_count = 0
    while p_state.buffer_ptr < len(sent.words) and step_count < 240:
      feats, label, next_state = oracle_step(p_state, sent, config)
      all_features.append(feats)
      all_labels.append(label)
      p_state = next_state
      step_count += 1

  # 5. Training Loop with Minibatches
  # Convert generated oracle instances to JNP arrays
  X_train = jnp.array(all_features)
  y_train = jnp.array(all_labels)

  output_path = "results/best_model.weights"
  best_uas = 0.0
  batch_size = 1024  # Standard batch size for this architecture
  print(f"Starting Training with {len(X_train)} instances...")

  # Initialize the dropout RNG once, then split within the loop
  rng = jax.random.PRNGKey(0)
  _, dropout_rng = jax.random.split(rng)

  for epoch in range(1, 11):
    epoch_losses = []

    # Use the generator for Stochastic Gradient Descent
    for batch_X, batch_y in get_minibatches(X_train, y_train, batch_size):
      # Re-split dropout RNG for every step to ensure stochasticity
      _, dropout_rng = jax.random.split(dropout_rng)

      # Apply the JIT-compiled training step
      state, loss = train_step(state, batch_X, batch_y, dropout_rng)
      epoch_losses.append(loss)

    # eval phase
    current_uas = calculate_uas(state.params, dev_sentences, model, config)
    avg_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Dev UAS: {current_uas * 100:.2f}%")

    # save the best model
    if current_uas > best_uas:
      best_uas = current_uas
      save_params(state, output_path)
      print("New best UAS achieved!")

  # final testing
  print("\nRestoring best model for final testing...")
  state = load_params(state, output_path)
  test_uas = calculate_uas(state.params, test_sentences, model, config)
  print(f"Final Test UAS: {test_uas * 100:.2f}%")


if __name__ == "__main__":
  main()

```


# `schema.py` (python)

```python
from typing import NamedTuple, Dict
import jax.numpy as jnp


class Sentence(NamedTuple):
  """A single sentence represented as fixed-width integer sequences."""

  words: jnp.ndarray  # Shape: (max_seq_len,)
  pos: jnp.ndarray  # Shape: (max_seq_len,)
  heads: jnp.ndarray  # Shape: (max_seq_len,)
  labels: jnp.ndarray  # Shape: (max_seq_len,)
  mask: jnp.ndarray  # Boolean mask for padding


class ParserState(NamedTuple):
  """The immutable state of a transition-based parser."""

  stack: jnp.ndarray  # Indices of words on stack
  buffer: jnp.ndarray  # Indices of words in buffer
  dependencies: jnp.ndarray  # (head, dependent, label_id) triples
  stack_ptr: int  # Current top of stack
  buffer_ptr: int  # Current front of buffer
  num_deps: int  # Counter for dependencies


class ParserVocab(NamedTuple):
  """Mappings for string-to-ID conversions."""

  word2id: Dict[str, int]
  pos2id: Dict[str, int]
  label2id: Dict[str, int]
  id2label: Dict[int, str]

```


# `utils.py` (python)

```python
import pickle
import os


def save_params(state, path):
  """Saves the model parameters to a file."""
  # Ensure directory exists
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    # We only save state.params to keep the file lightweight
    pickle.dump(state.params, f)
  print(f"Model parameters saved to {path}")


def load_params(state, path):
  """Loads parameters from a file into the current TrainState."""
  with open(path, "rb") as f:
    params = pickle.load(f)
  # Returns a new state with updated parameters
  return state.replace(params=params)

```


