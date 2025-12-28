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
  """
  Calculates the Unlabeled Attachment Score (UAS) on a dataset.
  Pure function that compares predicted heads against gold heads.
  """
  total_correct = 0
  total_tokens = 0

  # Process the dev set in minibatches
  for i in range(0, len(dev_sentences), batch_size):
    batch = dev_sentences[i : i + batch_size]

    # Initialize states for the entire batch
    # max_stack and max_buffer should match your config/data_loader limits
    states = [init_state(len(s.words), 30, 120) for s in batch]
    # In JAX, we stack these into a single Pytree of arrays
    states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states)

    # Iterative parsing loop
    while not is_finished(states):
      states = minibatch_parse_step(params, states, batch, model, config)

    # Compare dependencies with gold heads
    # states.dependencies shape: (batch_size, max_deps, 2) -> [head, dependent]
    for b_idx, sent in enumerate(batch):
      # Extract predicted heads for this sentence
      predicted_heads = jnp.full(len(sent.words), -1)
      deps = states.dependencies[b_idx]

      # Fill predicted_heads array based on dependencies
      for head, dep in deps:
        if dep != -1:
          predicted_heads = predicted_heads.at[dep].set(head)

      # Calculate correct attachments, ignoring padding and ROOT
      # sent.mask identifies real tokens in the padded sequence
      gold_heads = sent.heads
      mask = sent.mask & (jnp.arange(len(sent.words)) > 0)  # Ignore ROOT at index 0

      correct = jnp.sum((predicted_heads == gold_heads) & mask)
      total_correct += correct
      total_tokens += jnp.sum(mask)

  return float(total_correct) / float(total_tokens)
