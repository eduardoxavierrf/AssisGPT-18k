import numpy as np

class Linear:
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = bias

    limit = np.sqrt(6 / (in_features + out_features), dtype=np.float32)
    self.weight = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
    
    if self.use_bias:
      self.bias = np.zeros(out_features, dtype=np.float32)
      self.grad_bias = np.zeros_like(self.bias, dtype=np.float32)
    else:
      self.bias = None
      self.grad_bias = None

    self.grad_weight = np.zeros_like(self.weight, dtype=np.float32)

  def forward(self, x):
    self.input = x
    out = np.matmul(x, self.weight)
    
    if self.use_bias:
      out += self.bias
    
    return out

  def backward(self, grad_output):
    self.grad_weight = np.einsum('bij,bik->jk', self.input, grad_output)
    
    if self.use_bias:
      self.grad_bias = np.sum(grad_output, axis=(0, 1))

    grad_input = np.matmul(grad_output, self.weight.T)
    return grad_input

  def update_params(self, lr):
    self.weight -= lr * self.grad_weight
    
    if self.use_bias:
      self.bias -= lr * self.grad_bias
  
  def get_params(self):
    params = {'weight': self.weight.tolist()}

    if self.use_bias:
      params['bias'] = self.bias.tolist()

    return params
  
  def set_params(self, params):
    self.weight = np.array(params['weight'], dtype=np.float32)
    self.grad_weight = np.zeros_like(self.weight, dtype=np.float32)

    if self.use_bias:
      self.bias = np.array(params['bias'], dtype=np.float32)
      self.grad_bias = np.zeros_like(self.bias, dtype=np.float32)