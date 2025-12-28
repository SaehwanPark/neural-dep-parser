# Neural Dependency Parser (JAX/Flax)

This project implements a transition-based neural dependency parser based on the Stanford architecture. It uses **JAX** for high-performance vectorized operations and **Flax** for the neural network implementation.

## 1. Project Structure

The project follows a functional programming (FP) paradigm where the parser state is treated as an immutable Pytree.

```text
.
├── .env                # Stores data locations (DATA_PATH=./data)
├── schema.py           # Immutable NamedTuples (Sentence, ParserState)
├── config.py           # Vocabulary offsets and special token IDs
├── data_loader.py      # CoNLL parsing and data vectorization
├── engine.py           # State transitions and feature extraction
├── oracle.py           # Gold action logic for supervised training
├── parser_model.py     # Flax-based Feed-Forward Neural Network
├── inference.py        # Batched greedy parsing via jax.vmap
├── utils.py            # Model checkpointing (saving/loading)
└── run.py              # Main training and evaluation orchestrator

```

---

## 2. Environment Setup

This project uses `uv` for lightning-fast dependency management.

```bash
# Initialize the project environment
uv init

# Add required JAX/Flax dependencies
uv add jax jaxlib flax optax python-dotenv numpy tqdm

# Create the data directory and ensure CoNLL files are present
mkdir data
# Place train.conll, dev.conll, and test.conll in ./data/

```

Create a `.env` file in the root directory:

```text
DATA_PATH=./data

```

---

## 3. How it Works

### The Functional Engine

Unlike traditional parsers that mutate objects, this engine uses pure transformations:

* **`apply_transition(state, action)`**: Uses `jax.lax.switch` and `at[].set()` to return a brand new `ParserState`.
* **`extract_features(state, sentence)`**: Extracts 36 linguistic features (18 word IDs, 18 POS IDs) using JAX indexing.

### Vectorized Inference

Evaluation is accelerated using `jax.vmap`, allowing the parser to process hundreds of sentences in parallel on GPU/TPU.

---

## 4. Running the Parser

### Training

Execute the training loop, which will automatically generate oracle instances, train the model, and evaluate UAS on the development set:

```bash
uv run python run.py

```

### Evaluation Metric

The parser uses **Unlabeled Attachment Score (UAS)**, calculating the percentage of words correctly attached to their heads.

* **Gold Heads**: Extracted from the `heads` field in the `Sentence` Pytree.
* **Masking**: Padding and ROOT tokens are ignored during scoring.

---

## 5. Model Checkpoints

The script automatically saves the parameters of the best-performing model (highest Dev UAS) to `results/best_model.weights`. These can be reloaded for final testing or deployment.
