import jax.numpy as jnp
from schema import ParserState, Sentence
from engine import extract_features, apply_transition


def get_gold_transition(state: ParserState, sentence: Sentence) -> int:
  """
  Determines the correct transition to take based on the gold heads.
  Returns: 0 (LA), 1 (RA), or 2 (S).
  """
  if state.stack_ptr < 1:  # Only ROOT on stack
    return 2  # SHIFT

  # Indices of the top two elements on the stack
  s1 = state.stack[state.stack_ptr]  # top
  s2 = state.stack[state.stack_ptr - 1]  # second top

  # Check if s2's head is s1 (Left-Arc)
  # Note: index 0 is ROOT, so we don't Left-Arc the ROOT
  if s2 > 0 and sentence.heads[s2] == s1:
    return 0  # LEFT-ARC

  # Check if s1's head is s2 (Right-Arc)
  # AND ensure s1 has no more dependents left in the buffer
  if sentence.heads[s1] == s2:
    # Check buffer for any word whose head is s1
    has_buffer_dependents = jnp.any(sentence.heads[state.buffer_ptr :] == s1)
    if not has_buffer_dependents:
      return 1  # RIGHT-ARC

  return 2  # SHIFT


def oracle_step(state: ParserState, sentence: Sentence, config) -> tuple:
  """
  A single step of the oracle used for generating training instances.
  """
  features = extract_features(state, sentence, config)
  gold_action = get_gold_transition(state, sentence)
  next_state = apply_transition(state, gold_action)
  return features, gold_action, next_state
