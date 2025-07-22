import numpy as np

class Embedding:
  def __init__(self, num_embed: int, embed_dim: int) -> None:
    self.num_embed = num_embed
    self.embed_dim = embed_dim
    self.weight = np.random.randn(num_embed, embed_dim).astype(np.float32) * 0.01
    self.inputs: np.ndarray | None = None
    self.dW = np.zeros_like(self.weight, dtype=np.float32)

  def forward(self, tokens: np.ndarray) -> np.ndarray:
    self.inputs = tokens

    return self.weight[tokens]
  
  def backward(self, dout: np.ndarray) -> None:
    self.dW.fill(0)
    np.add.at(self.dW, self.inputs, dout)

  def update_params(self, learning_rate: float) -> None:
    self.weight -= learning_rate * self.dW

  def get_params(self):
    return {
      'weight': self.weight.tolist()
    }

  def set_params(self, params):
    self.weight = np.array(params['weight'], dtype=np.float32)
    self.dW = np.zeros_like(self.weight, dtype=np.float32)