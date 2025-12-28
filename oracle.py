import jax.numpy as jnp
from schema import ParserState, Sentence
from engine import extract_features, apply_transition


def get_gold_transition(state: ParserState, sentence: Sentence) -> int:
  # 0: Shift, 1: Left-Arc, 2: Right-Arc

  # If stack has only ROOT (or empty), we must SHIFT if possible
  if state.stack_ptr < 1:
    # If buffer is already empty, there's nothing sensible to do; default SHIFT.
    return 0

  s1 = state.stack[state.stack_ptr]  # top
  s2 = state.stack[state.stack_ptr - 1]  # second top

  # LEFT-ARC: s2 <- s1 (s1 is head of s2), and s2 cannot be ROOT
  if s2 > 0 and sentence.heads[s2] == s1:
    return 1

  # RIGHT-ARC: s2 -> s1 (s2 is head of s1), but only if s1 has no dependents left in buffer
  if sentence.heads[s1] == s2:
    # Remaining token indices in buffer (exclude sentinels)
    rem = state.buffer[state.buffer_ptr :]
    rem = rem[rem != -1]
    has_buffer_dependents = jnp.any(sentence.heads[rem] == s1)
    if not has_buffer_dependents:
      return 2

  return 0


def oracle_step(state: ParserState, sentence: Sentence, config) -> tuple:
  """
  A single step of the oracle used for generating training instances.
  """
  features = extract_features(state, sentence, config)
  gold_action = get_gold_transition(state, sentence)
  next_state = apply_transition(state, gold_action)
  return features, gold_action, next_state
