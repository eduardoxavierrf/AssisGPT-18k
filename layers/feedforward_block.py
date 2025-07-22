from .feedforward import FeedForward
from .layer_norm import LayerNorm

class FeedForwardBlock:
  def __init__(self, embed_dim, hidden_dim):
    self.ff = FeedForward(embed_dim, hidden_dim)
    self.norm = LayerNorm(embed_dim)

  def forward(self, x):
    self.input = x
    self.ff_out = self.ff.forward(x)
    self.out = self.norm.forward(x + self.ff_out)
    return self.out

  def backward(self, grad_output):
    grad_norm = self.norm.backward(grad_output)
    grad_ff = self.ff.backward(grad_norm)
    return grad_ff + grad_norm

  def update_params(self, lr):
    self.ff.update_params(lr)
    self.norm.update_params(lr)

  def get_params(self):
    return {
      'ff': self.ff.get_params(),
      'norm': self.norm.get_params()
    }

  def set_params(self, params):
    self.ff.set_params(params['ff'])
    self.norm.set_params(params['norm'])