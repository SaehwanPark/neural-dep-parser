from typing import NamedTuple, Dict
import jax.numpy as jnp


class Sentence(NamedTuple):
  """a single sentence represented as fixed-width integer sequences."""

  words: jnp.ndarray  # shape: (max_seq_len,)
  pos: jnp.ndarray  # shape: (max_seq_len,)
  heads: jnp.ndarray  # shape: (max_seq_len,)
  labels: jnp.ndarray  # shape: (max_seq_len,)
  mask: jnp.ndarray  # boolean mask for padding


class ParserState(NamedTuple):
  """the immutable state of a transition-based parser."""

  stack: jnp.ndarray  # indices of words on stack
  buffer: jnp.ndarray  # indices of words in buffer
  dependencies: jnp.ndarray  # (head, dependent, label_id) triples
  stack_ptr: int  # current top of stack
  buffer_ptr: int  # current front of buffer
  num_deps: int  # counter for dependencies


class ParserVocab(NamedTuple):
  """mappings for string-to-ID conversions."""

  word2id: Dict[str, int]
  pos2id: Dict[str, int]
  label2id: Dict[str, int]
  id2label: Dict[int, str]
