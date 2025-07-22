import numpy as np
import layers
import os
import json
from datetime import datetime

class Model:
  def __init__(self, vocab_size, embed_dim, context_len):
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.context_len = context_len

    self.layers = [
      layers.Embedding(vocab_size, embed_dim),
      layers.PositionalEncoding(context_len, embed_dim),
      layers.AttentionBlock(embed_dim),
      layers.FeedForwardBlock(embed_dim, 4 * embed_dim),
      layers.AttentionBlock(embed_dim),
      layers.FeedForwardBlock(embed_dim, 4 * embed_dim),
      layers.Linear(embed_dim, vocab_size)
    ]

  def forward(self, x):
    out = x
    for layer in self.layers:
      out = layer.forward(out)
    return out

  def backward(self, d_logits):
    grad = d_logits
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def compute_loss(self, logits, y_true):
    batch_size, context_len, vocab_size = logits.shape

    logits_flat = logits.reshape(-1, vocab_size)
    y_true_flat = y_true.flatten()

    logits_shifted = logits_flat - np.max(logits_flat, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[np.arange(len(y_true_flat)), y_true_flat] + 1e-9)
    loss = np.mean(correct_logprobs)

    d_logits = probs
    d_logits[np.arange(len(y_true_flat)), y_true_flat] -= 1
    d_logits /= batch_size * context_len
    d_logits = d_logits.reshape(batch_size, context_len, vocab_size)

    return loss, d_logits

  def update_params(self, learning_rate):
    for layer in self.layers:
      layer.update_params(learning_rate)

  def train_step(self, x_batch, y_batch, learning_rate):
    logits = self.forward(x_batch)
    loss, d_logits = self.compute_loss(logits, y_batch)
    self.backward(d_logits)
    self.update_params(learning_rate)
    return loss
  
  def predict(self, x: np.ndarray, temperature: float = 1.0, top_k: int = None) -> np.ndarray:
    logits = self.forward(x)
    last_logits = logits[:, -1, :]

    if temperature <= 0:
      raise ValueError("Temperature must be greater than 0")

    scaled_logits = last_logits / temperature

    if top_k is not None and top_k > 0:
      top_k_indices = np.argsort(scaled_logits, axis=-1)[:, -top_k:]
      mask = np.full_like(scaled_logits, -np.inf)
      batch_indices = np.arange(scaled_logits.shape[0])[:, None]
      mask[batch_indices, top_k_indices] = scaled_logits[batch_indices, top_k_indices]
      scaled_logits = mask

    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return np.array([
      np.random.choice(probs.shape[-1], p=probs[i])
      for i in range(probs.shape[0])
    ])

  def get_config(self):
    return {
      'vocab_size': self.vocab_size,
      'embed_dim': self.embed_dim,
      'context_len': self.context_len
    }
  
  def get_params(self):
    return [layer.get_params() for layer in self.layers]

  def set_params(self, params):
    for layer, param in zip(self.layers, params):
      layer.set_params(param)

  def save(self, path):
    os.makedirs(path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    timestamped_path = os.path.join(path, timestamp)
    os.makedirs(timestamped_path, exist_ok=True)
    
    config_path = os.path.join(timestamped_path, "config.json")
    with open(config_path, 'w') as f:
      json.dump(self.get_config(), f)
    
    params_path = os.path.join(timestamped_path, "weights.json")
    with open(params_path, 'w') as f:
      json.dump(self.get_params(), f)
    
    print(f"Model saved to {timestamped_path}")

  @staticmethod
  def load(path):
    config_path = os.path.join(path, "config.json")
    with open(config_path, 'r') as f:
      config = json.load(f)

    model = Model(**config)

    params_path = os.path.join(path, "weights.json")
    with open(params_path, 'r') as f:
      params = json.load(f)

    model.set_params(params)
    print(f"Model loaded from {path}")
    return model