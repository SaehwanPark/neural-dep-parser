import jax
import jax.numpy as jnp
from schema import ParserState, Sentence, ParserConfig


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

  # 1. Basic Stack and Buffer positions
  s0 = state.stack[state.stack_ptr]
  s1 = state.stack[state.stack_ptr - 1] if state.stack_ptr >= 1 else -1
  s2 = state.stack[state.stack_ptr - 2] if state.stack_ptr >= 2 else -1

  b0 = state.buffer[state.buffer_ptr] if state.buffer_ptr < len(state.buffer) else -1
  b1 = (
    state.buffer[state.buffer_ptr + 1]
    if state.buffer_ptr + 1 < len(state.buffer)
    else -1
  )
  b2 = (
    state.buffer[state.buffer_ptr + 2]
    if state.buffer_ptr + 2 < len(state.buffer)
    else -1
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
