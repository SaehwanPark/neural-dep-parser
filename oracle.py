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
