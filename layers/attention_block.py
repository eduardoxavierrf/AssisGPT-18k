from .attention import Attention
from .layer_norm import LayerNorm

class AttentionBlock:
  def __init__(self, embed_dim):
    self.attn = Attention(embed_dim, 2)
    self.norm = LayerNorm(embed_dim)

  def forward(self, x):
    self.input = x
    self.attn_out = self.attn.forward(x)
    self.out = self.norm.forward(x + self.attn_out)
    return self.out

  def backward(self, grad_out):
    grad_norm = self.norm.backward(grad_out)
    grad_attn = self.attn.backward(grad_norm)
    return grad_attn + grad_norm

  def update_params(self, lr):
    self.attn.update_params(lr)
    self.norm.update_params(lr)

  def get_params(self):
    return {
      'attn': self.attn.get_params(),
      'norm': self.norm.get_params()
    }

  def set_params(self, params):
    self.attn.set_params(params['attn'])
    self.norm.set_params(params['norm'])