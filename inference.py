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
  """
  Checks if all sentences in the batch are finished (buffer exhausted).
  """
  buf = states.buffer
  bptr = states.buffer_ptr
  max_len = buf.shape[1]

  clipped = jnp.minimum(bptr, max_len - 1)
  next_tok = jnp.take_along_axis(buf, clipped[:, None], axis=1)[:, 0]

  done = (bptr >= max_len) | (next_tok == -1)
  return jnp.all(done)


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
    states_list = [init_state(int(jnp.sum(s.mask)) + 1, 30, 120) for s in batch_list]
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
