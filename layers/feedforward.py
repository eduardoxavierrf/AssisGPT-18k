import numpy as np
from .linear import Linear

class FeedForward:
  def __init__(self, embed_dim, hidden_dim):
    self.linear1 = Linear(embed_dim, hidden_dim)
    self.linear2 = Linear(hidden_dim, embed_dim)

  def forward(self, x):
    self.input = x
    self.hidden = np.maximum(0, self.linear1.forward(x))
    return self.linear2.forward(self.hidden)

  def backward(self, grad_output):
    grad_hidden = self.linear2.backward(grad_output)

    relu_grad = grad_hidden * (self.hidden > 0)

    return self.linear1.backward(relu_grad)

  def update_params(self, lr):
    self.linear1.update_params(lr)
    self.linear2.update_params(lr)

  def get_params(self):
    return {
      'linear1': self.linear1.get_params(),
      'linear2': self.linear2.get_params()
    }

  def set_params(self, params):
    self.linear1.set_params(params['linear1'])
    self.linear2.set_params(params['linear2'])