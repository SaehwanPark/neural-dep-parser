import jax
import jax.numpy as jnp
from engine import extract_features, apply_transition, predict_action, init_state
from config import MAX_STACK_SIZE, MAX_BUFFER_SIZE


def minibatch_parse_step(params, states, sentences, model, config):
  """
  performs one parsing step for a batch of sentences.
  uses jax.vmap to parallelize across the batch dimension.
  """

  # 1. vectorized feature extraction
  def get_feats(state, sent):
    return extract_features(state, sent, config)

  batch_feats = jax.vmap(get_feats)(states, sentences)

  # 2. vectorized model prediction
  logits = model.apply({"params": params}, batch_feats, train=False)

  # 3. vectorized action selection and state update
  def update_single_state(ll, s):
    action = predict_action(ll, s)
    return apply_transition(s, action)

  new_states = jax.vmap(update_single_state)(logits, states)
  return new_states


def is_finished(states):
  """
  finished when:
    - buffer is empty (next token is -1 or ptr out of bounds)
    - and stack reduced to ROOT only (stack_ptr == 0)
  """
  buf = states.buffer
  bptr = states.buffer_ptr
  max_len = buf.shape[1]

  clipped = jnp.minimum(bptr, max_len - 1)
  next_tok = jnp.take_along_axis(buf, clipped[:, None], axis=1)[:, 0]

  buffer_empty = (bptr >= max_len) | (next_tok == -1)
  stack_done = states.stack_ptr == 0
  return jnp.all(buffer_empty & stack_done)


def calculate_uas(params, dev_sentences, model, config, batch_size: int = 1024):
  """
  calculates unlabeled attachment score on development/test set.

  FIXED: uses exact max steps calculation (2n transitions for sentence length n)
  instead of magic number.
  """
  total_correct = 0
  total_tokens = 0

  for i in range(0, len(dev_sentences), batch_size):
    batch_list = dev_sentences[i : i + batch_size]
    batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *batch_list)

    states_list = [
      init_state(int(jnp.sum(s.mask)) + 1, MAX_STACK_SIZE, MAX_BUFFER_SIZE)
      for s in batch_list
    ]
    states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states_list)

    # exact max steps: theoretical maximum is 2n (n shifts + n arcs) for sentence length n
    max_steps = 2 * batch.words.shape[1]
    steps = 0

    while not is_finished(states) and steps < max_steps:
      states = minibatch_parse_step(params, states, batch, model, config)
      steps += 1

    for batch_idx in range(len(batch_list)):
      sent = batch_list[batch_idx]
      predicted_heads = jnp.full(len(sent.words), -1, dtype=jnp.int32)
      deps = states.dependencies[batch_idx]

      for head, dep in deps:
        if dep != -1:
          predicted_heads = predicted_heads.at[dep].set(head)

      gold_heads = sent.heads
      mask = sent.mask & (jnp.arange(len(sent.words)) > 0)

      total_correct += jnp.sum((predicted_heads == gold_heads) & mask)
      total_tokens += jnp.sum(mask)

  return float(total_correct) / float(total_tokens)
