import numpy as np

class LayerNorm:
  def __init__(self, dim, eps=1e-5):
    self.eps = eps
    self.gamma = np.ones(dim, dtype=np.float32)
    self.beta = np.zeros(dim, dtype=np.float32)

  def forward(self, x):
    self.input = x
    self.mean = x.mean(axis=-1, keepdims=True)
    self.var = x.var(axis=-1, keepdims=True)
    self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
    return self.gamma * self.norm + self.beta

  def backward(self, grad_output):
    N = self.input.shape[-1]

    std_inv = 1.0 / np.sqrt(self.var + self.eps)
    x_mu = self.input - self.mean

    dnorm = grad_output * self.gamma

    dvar = np.sum(dnorm * x_mu * -0.5 * std_inv**3, axis=-1, keepdims=True)
    dmean = np.sum(dnorm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=-1, keepdims=True)

    grad_input = dnorm * std_inv + dvar * 2 * x_mu / N + dmean / N

    axes_to_sum = tuple(range(len(grad_output.shape) - 1))
    
    self.grad_gamma = np.sum(grad_output * self.norm, axis=axes_to_sum)
    self.grad_beta = np.sum(grad_output, axis=axes_to_sum)

    return grad_input

  def update_params(self, lr):
    self.gamma -= lr * self.grad_gamma
    self.beta -= lr * self.grad_beta

  def get_params(self):
    return {
      'gamma': self.gamma.tolist(),
      'beta': self.beta.tolist()
    }

  def set_params(self, params):
    self.gamma = np.array(params['gamma'], dtype=np.float32)
    self.beta = np.array(params['beta'], dtype=np.float32)
    self.grad_gamma = np.zeros_like(self.gamma, dtype=np.float32)
    self.grad_beta = np.zeros_like(self.beta, dtype=np.float32)