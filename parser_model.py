import flax.linen as nn


class ParserModel(nn.Module):
  """
  flax implementation of the feed-forward dependency parser.
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
    # 1. custom embedding lookup (manual implementation as per requirement)
    # we use a learnable parameter for embeddings
    embeddings = self.param(
      "embeddings",
      nn.initializers.uniform(scale=0.1),
      (self.vocab_size, self.embed_size),
    )

    # select embeddings and flatten: (batch, n_features * embed_size)
    # equivalent to PyTorch's x.view(batch_size, -1)
    x = embeddings[x].reshape((x.shape[0], -1))

    # 2. hidden layer: affine -> relu -> dropout
    # we use xavier uniform (glorot) for weights
    x = nn.Dense(
      features=self.hidden_size,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.uniform(),
    )(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

    # 3. output layer (logits)
    logits = nn.Dense(
      features=self.n_classes,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.uniform(),
    )(x)

    return logits
