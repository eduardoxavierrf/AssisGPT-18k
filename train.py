import numpy as np
from model import Model
from data.tokenizer import CharTokenizer
from data.utils import calculate_accuracy, load_text, create_dataset
from datetime import datetime

def main():
  raw_text = load_text("data.txt")
  
  tokenizer = CharTokenizer.load('data/little_char_idx.json')

  encoded = np.array(tokenizer.encode(raw_text), dtype=np.int32)

  context_len = 8
  vocab_size = tokenizer.vocab_size
  embed_dim = 8

  x_train, y_train, x_val, y_val = create_dataset(encoded, context_len, train_split=0.8)

  model = Model(vocab_size, embed_dim, context_len)

  epochs = 1
  batch_size = 32
  learning_rate = 0.05
  print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
  print("Epoch | Train Loss | Train Acc |  Val Acc  | Time")
  print("-" * 45)

  for epoch in range(epochs):
    total_loss = 0
    num_batches = x_train.shape[0] // batch_size
    for i in range(num_batches):
      x_batch = x_train[i*batch_size:(i+1)*batch_size]
      y_batch = y_train[i*batch_size:(i+1)*batch_size]
      loss = model.train_step(x_batch, y_batch, learning_rate)
      total_loss += loss

    avg_loss = total_loss / num_batches
    train_accuracy = calculate_accuracy(model, x_train, y_train, batch_size)
    val_accuracy = calculate_accuracy(model, x_val, y_val, batch_size)

    now = datetime.now().strftime("%H:%M:%S")
    print(f"{epoch+1:5d} | {avg_loss:10.4f} | {train_accuracy:9.4f} | {val_accuracy:9.4f} | {now}")

  model.save("models")

if __name__ == "__main__":
  main()