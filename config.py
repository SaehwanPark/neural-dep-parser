from typing import NamedTuple
from schema import ParserVocab

# extracted constants from magic numbers
MAX_STACK_SIZE = 30
MAX_BUFFER_SIZE = 120
MAX_ORACLE_STEPS = 400  # conservative upper bound for oracle trajectories
MAX_PARSE_STEPS = 400  # conservative upper bound for inference


class ParserConfig(NamedTuple):
  """constants and special IDs for the parser logic."""

  n_features: int = 36
  hidden_size: int = 200
  n_classes: int = 3
  embed_size: int = 50
  dropout_rate: float = 0.5

  # special IDs mapped from ParserVocab
  NULL_ID: int = 0
  P_NULL_ID: int = 0
  ROOT_ID: int = 0
  P_ROOT_ID: int = 0
  UNK_ID: int = 0
  P_UNK_ID: int = 0


def create_config(vocab: ParserVocab) -> ParserConfig:
  """factory function to populate IDs based on the actual vocab with validation."""
  # validate required tokens exist
  required_word_tokens = ["<NULL>", "<ROOT>", "<UNK>"]
  required_pos_tokens = ["<p>:<NULL>", "<p>:<ROOT>", "<p>:<UNK>"]

  for tok in required_word_tokens:
    if tok not in vocab.word2id:
      raise ValueError(f"missing required word token in vocabulary: {tok}")

  for tok in required_pos_tokens:
    if tok not in vocab.pos2id:
      raise ValueError(f"missing required POS token in vocabulary: {tok}")

  return ParserConfig(
    NULL_ID=vocab.word2id["<NULL>"],
    P_NULL_ID=vocab.pos2id["<p>:<NULL>"],
    ROOT_ID=vocab.word2id["<ROOT>"],
    P_ROOT_ID=vocab.pos2id["<p>:<ROOT>"],
    UNK_ID=vocab.word2id["<UNK>"],
    P_UNK_ID=vocab.pos2id["<p>:<UNK>"],
  )
