import jax.numpy as jnp
from schema import ParserState, Sentence
from engine import extract_features, apply_transition


def get_gold_transition(state: ParserState, sentence: Sentence) -> int:
  """
  0: SHIFT, 1: LEFT-ARC, 2: RIGHT-ARC
  """
  # can we shift?
  can_shift = (state.buffer_ptr < state.buffer.shape[0]) and (
    state.buffer[state.buffer_ptr] != -1
  )

  # if stack too small, must shift if possible; otherwise reduce
  if state.stack_ptr < 1:
    return 0 if can_shift else 2

  s1 = state.stack[state.stack_ptr]
  s2 = state.stack[state.stack_ptr - 1]

  # LEFT-ARC: s2 <- s1 (s1 is head of s2), s2 cannot be ROOT
  if s2 > 0 and sentence.heads[s2] == s1:
    return 1

  # RIGHT-ARC: s2 -> s1 (s2 is head of s1), only if s1 has no dependents remaining in buffer
  if sentence.heads[s1] == s2:
    rem = state.buffer[state.buffer_ptr :]
    rem = rem[rem != -1]
    has_buffer_dependents = jnp.any(sentence.heads[rem] == s1)
    if not has_buffer_dependents:
      return 2

  # otherwise shift if possible, else reduce (cleanup)
  return 0 if can_shift else 2


def oracle_step(state: ParserState, sentence: Sentence, config) -> tuple:
  """
  A single step of the oracle used for generating training instances.
  """
  features = extract_features(state, sentence, config)
  gold_action = get_gold_transition(state, sentence)
  next_state = apply_transition(state, gold_action)
  return features, gold_action, next_state
