import pickle
import os


def save_params(state, path):
  """Saves the model parameters to a file."""
  # Ensure directory exists
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    # We only save state.params to keep the file lightweight
    pickle.dump(state.params, f)
  print(f"Model parameters saved to {path}")


def load_params(state, path):
  """Loads parameters from a file into the current TrainState."""
  with open(path, "rb") as f:
    params = pickle.load(f)
  # Returns a new state with updated parameters
  return state.replace(params=params)
