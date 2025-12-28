import flax.linen as nn


class ParserModel(nn.Module):
  """
  Flax implementation of the Feed-Forward Dependency Parser.
  """

  vocab_size: int
  embed_size: int = 50
  hidden_size: int = 200
  n_classes: int = 3
  dropout_rate: float = 0.5

  @nn.compact
  def __call__(self, x, train: bool = True):
    """
    x: (batch_size, n_features) - indices of features
    """
    # 1. Custom Embedding Lookup (Manual implementation as per requirement)
    # We use a learnable parameter for embeddings
    embeddings = self.param(
      "embeddings",
      nn.initializers.uniform(scale=0.1),
      (self.vocab_size, self.embed_size),
    )

    # Select embeddings and flatten: (batch, n_features * embed_size)
    # Equivalent to PyTorch's x.view(batch_size, -1)
    x = embeddings[x].reshape((x.shape[0], -1))

    # 2. Hidden Layer: Affine -> ReLU -> Dropout
    # We use Xavier Uniform (Glorot) for weights
    x = nn.Dense(
      features=self.hidden_size,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.uniform(),
    )(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

    # 3. Output Layer (Logits)
    logits = nn.Dense(
      features=self.n_classes,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.uniform(),
    )(x)

    return logits
