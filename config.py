from typing import NamedTuple
from schema import ParserVocab


class ParserConfig(NamedTuple):
  """Constants and special IDs for the parser logic."""

  n_features: int = 36
  hidden_size: int = 200
  n_classes: int = 3
  embed_size: int = 50
  dropout_rate: float = 0.5

  # Special IDs mapped from ParserVocab
  NULL_ID: int = 0
  P_NULL_ID: int = 0
  ROOT_ID: int = 0
  P_ROOT_ID: int = 0
  UNK_ID: int = 0
  P_UNK_ID: int = 0


def create_config(vocab: ParserVocab) -> ParserConfig:
  """Factory function to populate IDs based on the actual vocab."""
  return ParserConfig(
    NULL_ID=vocab.word2id["<NULL>"],
    P_NULL_ID=vocab.pos2id["<p>:<NULL>"],
    ROOT_ID=vocab.word2id["<ROOT>"],
    P_ROOT_ID=vocab.pos2id["<p>:<ROOT>"],
    UNK_ID=vocab.word2id["<UNK>"],
    P_UNK_ID=vocab.pos2id["<p>:<UNK>"],
  )
