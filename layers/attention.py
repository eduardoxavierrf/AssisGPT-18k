import numpy as np
from .linear import Linear

class Attention:
  def __init__(self, embed_dim, num_heads):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    self.q_proj = Linear(embed_dim, embed_dim, bias=False)
    self.k_proj = Linear(embed_dim, embed_dim, bias=False)
    self.v_proj = Linear(embed_dim, embed_dim, bias=False)
    self.out_proj = Linear(embed_dim, embed_dim, bias=False)

    self._causal_mask = None
    self._mask_size = 0

  def forward(self, x):
    self.x = x
    batch_size, seq_len, _ = x.shape

    Q = self.q_proj.forward(x)
    K = self.k_proj.forward(x)
    V = self.v_proj.forward(x)

    Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    self.Q, self.K, self.V = Q, K, V

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
    scores = self._apply_causal_mask(scores, seq_len)

    self.attn_weights = self._softmax(scores)
    attn_output = self.attn_weights @ V

    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
    self.attn_output = attn_output

    return self.out_proj.forward(attn_output)

  def backward(self, grad_out):
    batch_size, seq_len, _ = grad_out.shape

    grad_attn_output = self.out_proj.backward(grad_out)

    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    grad_attn_weights = grad_attn_output @ self.V.transpose(0, 1, 3, 2)
    grad_V = self.attn_weights.transpose(0, 1, 3, 2) @ grad_attn_output

    grad_scores = self._softmax_backward(self.attn_weights, grad_attn_weights)
    causal_mask = self._get_causal_mask(seq_len)
    grad_scores = grad_scores * causal_mask[None, None, :, :]

    grad_Q = grad_scores @ self.K / np.sqrt(self.head_dim)
    grad_K = grad_scores.transpose(0, 1, 3, 2) @ self.Q / np.sqrt(self.head_dim)

    grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
    grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
    grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

    grad_input_Q = self.q_proj.backward(grad_Q)
    grad_input_K = self.k_proj.backward(grad_K)
    grad_input_V = self.v_proj.backward(grad_V)

    return grad_input_Q + grad_input_K + grad_input_V

  def update_params(self, lr):
    self.q_proj.update_params(lr)
    self.k_proj.update_params(lr)
    self.v_proj.update_params(lr)
    self.out_proj.update_params(lr)

  def _get_causal_mask(self, seq_len):
    if self._causal_mask is None or self._mask_size < seq_len:
      self._causal_mask = np.tril(np.ones((seq_len, seq_len)))
      self._mask_size = seq_len
    return self._causal_mask[:seq_len, :seq_len]

  def _apply_causal_mask(self, scores, seq_len):
    causal_mask = self._get_causal_mask(seq_len)
    return np.where(causal_mask[None, None, :, :], scores, -1e9)

  def _softmax(self, x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

  def _softmax_backward(self, softmax_out, grad_output):
    return grad_output * softmax_out - softmax_out * np.sum(grad_output * softmax_out, axis=-1, keepdims=True)
  
  def get_params(self):
    return {
      'q_proj': self.q_proj.get_params(),
      'k_proj': self.k_proj.get_params(),
      'v_proj': self.v_proj.get_params(),
      'out_proj': self.out_proj.get_params()
    }

  def set_params(self, params):
    self.q_proj.set_params(params['q_proj'])
    self.k_proj.set_params(params['k_proj'])
    self.v_proj.set_params(params['v_proj'])
    self.out_proj.set_params(params['out_proj'])