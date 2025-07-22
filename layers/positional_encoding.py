import numpy as np

class PositionalEncoding:
  def __init__(self, max_len: int, embedding_dim: int):
    self.max_len = max_len
    self.embedding_dim = embedding_dim
    self.pe = self._create_positional_encoding()

  def _create_positional_encoding(self) -> np.ndarray:
    pe = np.zeros((self.max_len, self.embedding_dim), dtype=np.float32)
    position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, self.embedding_dim, 2) * -(np.log(10000.0) / self.embedding_dim), dtype=np.float32)

    pe[:, 0::2] = np.sin(position * div_term, dtype=np.float32)
    pe[:, 1::2] = np.cos(position * div_term, dtype=np.float32)

    return pe

  def forward(self, x: np.ndarray) -> np.ndarray:
    seq_len = x.shape[1]
    return x + self.pe[:seq_len]

  def backward(self, dout: np.ndarray) -> np.ndarray:
    return dout

  def update_params(self, learning_rate: float):
    pass

  def get_params(self):
    return {}

  def set_params(self, params):
    pass