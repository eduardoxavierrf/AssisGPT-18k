import numpy as np

def load_text(path):
  with open(path, 'r', encoding='utf-8') as f:
    return f.read()

def create_dataset(encoded, context_len, train_split=0.8):
  x, y = [], []
  
  for i in range(len(encoded) - context_len):
    x.append(encoded[i:i+context_len])
    y.append(encoded[i+1:i+context_len+1])
  
  x = np.array(x)
  y = np.array(y)
  
  indices = np.arange(x.shape[0])
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]
  
  split_idx = int(len(x) * train_split)
  
  x_train, x_val = x[:split_idx], x[split_idx:]
  y_train, y_val = y[:split_idx], y[split_idx:]
  
  return x_train, y_train, x_val, y_val

def calculate_accuracy(model, x_data, y_data, batch_size=32):
  correct_predictions = 0
  total_predictions = 0
  
  num_batches = x_data.shape[0] // batch_size
  
  for i in range(num_batches):
    x_batch = x_data[i*batch_size:(i+1)*batch_size]
    y_batch = y_data[i*batch_size:(i+1)*batch_size]
    
    logits = model.forward(x_batch)
    
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    predictions = np.argmax(probs, axis=-1)
    
    correct_predictions += np.sum(predictions == y_batch)
    total_predictions += y_batch.size
  
  remaining = x_data.shape[0] % batch_size
  if remaining > 0:
    x_batch = x_data[-remaining:]
    y_batch = y_data[-remaining:]
    logits = model.forward(x_batch)
    
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    predictions = np.argmax(probs, axis=-1)
    correct_predictions += np.sum(predictions == y_batch)
    total_predictions += y_batch.size

  return correct_predictions / total_predictions