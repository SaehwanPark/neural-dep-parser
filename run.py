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

  model = ParserModel(
    vocab_size=len(vocab.word2id) + len(vocab.pos2id),
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

  for sent in train_sentences[:1000]:  # Using a subset for demonstration
    p_state = init_state(len(sent.words), 30, 120)
    # Iterate until buffer is empty
    while p_state.buffer_ptr < len(sent.words):
      feats, label, next_state = oracle_step(p_state, sent, config)
      all_features.append(feats)
      all_labels.append(label)
      p_state = next_state

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
