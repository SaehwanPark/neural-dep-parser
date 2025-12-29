import os
import logging
import numpy as np
import jax.numpy as jnp
from dotenv import load_dotenv
from schema import Sentence, ParserVocab
from typing import List, Dict

load_dotenv()


def load_conll_data(file_name: str, lowercase: bool = True) -> List[Dict]:
  """
  robust CoNLL(-U-ish) loader.
  - splits on any whitespace (tabs OR spaces)
  - flushes last sentence even if file doesn't end with a blank line
  - skips multiword tokens like 1-2
  """

  logger = logging.getLogger(__name__)

  data_path = os.getenv("DATA_PATH", "./data")
  full_path = os.path.join(data_path, file_name)

  examples: List[Dict] = []
  word: List[str] = []
  pos: List[str] = []
  head: List[int] = []
  label: List[str] = []

  def flush():
    nonlocal word, pos, head, label
    if word:
      examples.append({"word": word, "pos": pos, "head": head, "label": label})
      word, pos, head, label = [], [], [], []

  with open(full_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        flush()
        continue

      sp = line.split()  # whitespace-agnostic
      if len(sp) < 8:
        flush()
        continue

      tok_id = sp[0]
      if "-" in tok_id:
        continue

      w = sp[1].lower() if lowercase else sp[1]
      xpos = sp[4]
      h = int(sp[6])
      rel = sp[7]

      word.append(w)
      pos.append(xpos)
      head.append(h)
      label.append(rel)

  flush()
  logger.info("loaded %d sentences from %s", len(examples), full_path)
  return examples


def build_vocab(train_data: List[Dict]) -> ParserVocab:
  """
  builds vocabularies with disjoint ID ranges.
  IDs are contiguous and stable.
  """
  # 1) labels
  unique_labels = sorted(set(ll for ex in train_data for ll in ex["label"]))
  label2id = {f"<l>:{ll}": i for i, ll in enumerate(unique_labels)}
  label2id["<l>:<NULL>"] = len(label2id)
  id2label = {i: ll for ll, i in label2id.items()}

  # 2) POS
  pos_offset = max(label2id.values()) + 1
  unique_pos = sorted(set(f"<p>:{p}" for ex in train_data for p in ex["pos"]))
  pos2id = {p: pos_offset + i for i, p in enumerate(unique_pos)}
  next_pos_id = max(pos2id.values(), default=pos_offset - 1) + 1
  pos2id["<p>:<UNK>"] = next_pos_id
  pos2id["<p>:<NULL>"] = next_pos_id + 1
  pos2id["<p>:<ROOT>"] = next_pos_id + 2

  # 3) words
  word_offset = max(pos2id.values()) + 1
  unique_words = sorted(set(w for ex in train_data for w in ex["word"]))
  word2id = {w: word_offset + i for i, w in enumerate(unique_words)}
  next_word_id = max(word2id.values(), default=word_offset - 1) + 1
  word2id["<UNK>"] = next_word_id
  word2id["<NULL>"] = next_word_id + 1
  word2id["<ROOT>"] = next_word_id + 2

  return ParserVocab(word2id, pos2id, label2id, id2label)


def vectorize_sentences(
  raw_data: List[Dict], vocab: ParserVocab, max_len: int = 120
) -> List[Sentence]:
  """
  indexing invariant:
  - position 0 is ROOT
  - real tokens are in positions 1..n (matching CoNLL token IDs)
  - mask is True for 1..n only (False for ROOT and padding)
  """
  sentences: List[Sentence] = []

  null_w = vocab.word2id["<NULL>"]
  null_p = vocab.pos2id["<p>:<NULL>"]
  root_w = vocab.word2id["<ROOT>"]
  root_p = vocab.pos2id["<p>:<ROOT>"]

  for ex in raw_data:
    n = len(ex["word"])
    n_store = min(n, max_len - 1)

    words = np.full(max_len, null_w, dtype=np.int32)
    pos = np.full(max_len, null_p, dtype=np.int32)
    heads = np.full(max_len, -1, dtype=np.int32)

    # ROOT
    words[0] = root_w
    pos[0] = root_p
    heads[0] = -1

    # tokens 1..n_store
    for i in range(n_store):
      j = i + 1
      words[j] = vocab.word2id.get(ex["word"][i], vocab.word2id["<UNK>"])
      pos_key = f"<p>:{ex['pos'][i]}"
      pos[j] = vocab.pos2id.get(pos_key, vocab.pos2id["<p>:<UNK>"])
      heads[j] = ex["head"][i]  # CoNLL heads already use 0 as ROOT

    mask = np.zeros(max_len, dtype=bool)
    mask[1 : (n_store + 1)] = True

    sentences.append(
      Sentence(
        words=jnp.array(words),
        pos=jnp.array(pos),
        heads=jnp.array(heads),
        labels=jnp.zeros(max_len, dtype=jnp.int32),
        mask=jnp.array(mask),
      )
    )

  return sentences
