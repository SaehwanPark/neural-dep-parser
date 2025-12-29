import os
import pickle
import logging

logger = logging.getLogger(__name__)


def save_params(state, path: str) -> None:
  """saves model parameters to a file."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    pickle.dump(state.params, f)
  logger.info("model parameters saved to %s", path)


def load_params(state, path: str):
  """loads parameters from a file into the current TrainState."""
  with open(path, "rb") as f:
    params = pickle.load(f)
  logger.info("model parameters loaded from %s", path)
  return state.replace(params=params)
