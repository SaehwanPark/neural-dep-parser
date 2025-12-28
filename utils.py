import os
import pickle
import logging

logger = logging.getLogger(__name__)


def save_params(state, path: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    pickle.dump(state.params, f)
  logger.info("Model parameters saved to %s", path)


def load_params(state, path: str):
  with open(path, "rb") as f:
    params = pickle.load(f)
  logger.info("Model parameters loaded from %s", path)
  return state.replace(params=params)
