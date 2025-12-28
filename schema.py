from typing import NamedTuple, Dict
import jax.numpy as jnp


class Sentence(NamedTuple):
  """A single sentence represented as fixed-width integer sequences."""

  words: jnp.ndarray  # Shape: (max_seq_len,)
  pos: jnp.ndarray  # Shape: (max_seq_len,)
  heads: jnp.ndarray  # Shape: (max_seq_len,)
  labels: jnp.ndarray  # Shape: (max_seq_len,)
  mask: jnp.ndarray  # Boolean mask for padding


class ParserState(NamedTuple):
  """The immutable state of a transition-based parser."""

  stack: jnp.ndarray  # Indices of words on stack
  buffer: jnp.ndarray  # Indices of words in buffer
  dependencies: jnp.ndarray  # (head, dependent, label_id) triples
  stack_ptr: int  # Current top of stack
  buffer_ptr: int  # Current front of buffer
  num_deps: int  # Counter for dependencies


class ParserVocab(NamedTuple):
  """Mappings for string-to-ID conversions."""

  word2id: Dict[str, int]
  pos2id: Dict[str, int]
  label2id: Dict[str, int]
  id2label: Dict[int, str]
