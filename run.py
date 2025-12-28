import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import logging

from dotenv import load_dotenv
from flax.training import train_state

from config import create_config
from data_loader import load_conll_data, build_vocab, vectorize_sentences
from parser_model import ParserModel
from engine import init_state
from oracle import oracle_step
from inference import calculate_uas
from utils import save_params, load_params

logger = logging.getLogger(__name__)


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
  logger = logging.getLogger(__name__)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  load_dotenv()
  data_path = os.getenv("DATA_PATH", "/home/saehwan/data/dep_parser")
  logger.info("Loading data from %s...", data_path)

  raw_train = load_conll_data("train.conll")
  vocab = build_vocab(raw_train)
  config = create_config(vocab)

  train_sentences = vectorize_sentences(raw_train, vocab)

  raw_dev = load_conll_data("dev.conll")
  dev_sentences = vectorize_sentences(raw_dev, vocab)

  raw_test = load_conll_data("test.conll")
  test_sentences = vectorize_sentences(raw_test, vocab)

  logger.info(
    "Train sentences: %d | Dev: %d | Test: %d",
    len(train_sentences),
    len(dev_sentences),
    len(test_sentences),
  )

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

  logger.info("Generating training instances via Oracle...")
  all_features = []
  all_labels = []

  # Optional cap to avoid OOM while debugging:
  # cap = int(os.getenv("TRAIN_SENT_CAP", "0"))
  # iterable = train_sentences[:cap] if cap > 0 else train_sentences
  iterable = train_sentences

  for sent_idx, sent in enumerate(iterable):
    sent_len = int(jnp.sum(sent.mask)) + 1
    p_state = init_state(sent_len, 30, 120)

    step_count = 0
    while step_count < 400:
      # finished when buffer empty and stack reduced
      if p_state.buffer_ptr >= p_state.buffer.shape[0]:
        buffer_empty = True
      else:
        buffer_empty = p_state.buffer[p_state.buffer_ptr] == -1

      if buffer_empty and p_state.stack_ptr == 0:
        break

      feats, label, next_state = oracle_step(p_state, sent, config)
      all_features.append(np.array(feats, dtype=np.int32))
      all_labels.append(int(label))
      p_state = next_state
      step_count += 1

    if (sent_idx + 1) % 2000 == 0:
      logger.info(
        "Processed %d sentences; instances so far: %d", sent_idx + 1, len(all_labels)
      )

  X_train = jnp.array(np.stack(all_features, axis=0))
  y_train = jnp.array(np.array(all_labels, dtype=np.int32))

  logger.info("Starting Training with %d instances...", int(X_train.shape[0]))

  output_path = "results/best_model.weights"
  best_uas = 0.0
  batch_size = 1024

  rng = jax.random.PRNGKey(0)
  _, dropout_rng = jax.random.split(rng)

  for epoch in range(1, 11):
    epoch_losses = []

    for batch_X, batch_y in get_minibatches(X_train, y_train, batch_size):
      _, dropout_rng = jax.random.split(dropout_rng)
      state, loss = train_step(state, batch_X, batch_y, dropout_rng)
      epoch_losses.append(loss)

    current_uas = calculate_uas(state.params, dev_sentences, model, config)
    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    logger.info(
      "Epoch %d | Loss: %.4f | Dev UAS: %.2f%%", epoch, avg_loss, current_uas * 100.0
    )

    if current_uas > best_uas:
      best_uas = current_uas
      save_params(state, output_path)
      logger.info("New best UAS achieved!")

  logger.info("Restoring best model for final testing...")
  state = load_params(state, output_path)
  test_uas = calculate_uas(state.params, test_sentences, model, config)
  logger.info("Final Test UAS: %.2f%%", test_uas * 100.0)


if __name__ == "__main__":
  main()
