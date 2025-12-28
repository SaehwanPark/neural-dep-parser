import os
import numpy as np
import jax.numpy as jnp
from dotenv import load_dotenv
from schema import Sentence, ParserVocab
from typing import List, Dict

load_dotenv()


def load_conll_data(file_name: str, lowercase: bool = True) -> List[Dict]:
  """
  Reads CoNLL file into a list of raw dictionaries.
  Pure function: no side effects other than file reading.
  """
  data_path = os.getenv("DATA_PATH", "./data")
  full_path = os.path.join(data_path, file_name)

  examples = []
  with open(full_path, "r", encoding="utf-8") as f:
    word, pos, head, label = [], [], [], []
    for line in f:
      sp = line.strip().split("\t")
      if len(sp) == 10:
        if "-" not in sp[0]:  # Skip multi-word tokens
          word.append(sp[1].lower() if lowercase else sp[1])
          pos.append(sp[4])
          head.append(int(sp[6]))
          label.append(sp[7])
      elif len(word) > 0:
        examples.append({"word": word, "pos": pos, "head": head, "label": label})
        word, pos, head, label = [], [], [], []
  return examples


def build_vocab(train_data: List[Dict]) -> ParserVocab:
  """Builds vocabularies with unique ID offsets for each feature type."""
  # 1. Labels (L_PREFIX)
  # root_labels = [
  #   ex["label"][i] for ex in train_data for i, h in enumerate(ex["head"]) if h == 0
  # ]
  # root_label = Counter(root_labels).most_common(1)[0][0]

  unique_labels = list(set([ll for ex in train_data for ll in ex["label"]]))
  label2id = {f"<l>:{ll}": i for i, ll in enumerate(unique_labels)}
  label2id["<l>:<NULL>"] = len(label2id)
  id2label = {i: ll for ll, i in label2id.items()}

  # 2. POS Tags (P_PREFIX) with offset
  pos_offset = len(label2id)
  all_pos = [f"<p>:{p}" for ex in train_data for p in ex["pos"]]
  unique_pos = list(set(all_pos))
  pos2id = {p: i + pos_offset for i, p in enumerate(unique_pos)}
  pos2id["<p>:<UNK>"] = len(pos2id) + pos_offset
  pos2id["<p>:<NULL>"] = len(pos2id) + pos_offset + 1
  pos2id["<p>:<ROOT>"] = len(pos2id) + pos_offset + 2

  # 3. Words with offset
  word_offset = pos_offset + len(pos2id)
  all_words = [w for ex in train_data for w in ex["word"]]
  unique_words = list(set(all_words))
  word2id = {w: i + word_offset for i, w in enumerate(unique_words)}
  word2id["<UNK>"] = len(word2id) + word_offset
  word2id["<NULL>"] = len(word2id) + word_offset + 1
  word2id["<ROOT>"] = len(word2id) + word_offset + 2

  return ParserVocab(word2id, pos2id, label2id, id2label)


def vectorize_sentences(
  raw_data: List[Dict], vocab: ParserVocab, max_len: int = 120
) -> List[Sentence]:
  """
  Converts raw string data into fixed-width jnp arrays.

  Indexing invariant after this:
  - position 0 is ROOT
  - real tokens are in positions 1..n (matching CoNLL token IDs)
  - mask is True for 1..n only (False for ROOT and padding)
  """
  sentences = []
  null_w = vocab.word2id["<NULL>"]
  null_p = vocab.pos2id["<p>:<NULL>"]
  root_w = vocab.word2id["<ROOT>"]
  root_p = vocab.pos2id["<p>:<ROOT>"]

  for ex in raw_data:
    n = len(ex["word"])
    # We store ROOT + up to (max_len - 1) tokens
    n_store = min(n, max_len - 1)

    words = np.full(max_len, null_w, dtype=np.int32)
    pos = np.full(max_len, null_p, dtype=np.int32)
    heads = np.full(max_len, -1, dtype=np.int32)

    # ROOT at 0
    words[0] = root_w
    pos[0] = root_p
    heads[0] = -1

    # Tokens at 1..n_store (so indices match CoNLL 1..n)
    for i in range(n_store):
      j = i + 1
      words[j] = vocab.word2id.get(ex["word"][i], vocab.word2id["<UNK>"])
      pos_key = f"<p>:{ex['pos'][i]}"
      pos[j] = vocab.pos2id.get(pos_key, vocab.pos2id["<p>:<UNK>"])
      heads[j] = ex["head"][i]  # CoNLL head indices already use 0=ROOT

    # mask True for real tokens only (exclude ROOT and padding)
    mask = np.zeros(max_len, dtype=bool)
    mask[1 : (n_store + 1)] = True

    sentences.append(
      Sentence(
        words=jnp.array(words),
        pos=jnp.array(pos),
        heads=jnp.array(heads),
        labels=jnp.zeros(max_len, dtype=np.int32),
        mask=jnp.array(mask),
      )
    )

  return sentences
