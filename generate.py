from model import Model
from data.tokenizer import CharTokenizer
import numpy as np

tokenizer = CharTokenizer.load("data/little_char_idx.json")

def generate_text_streaming(model, tokenizer, prompt, max_new_tokens=100):
  model_input = np.array([tokenizer.encode(prompt)], dtype=int)

  print(prompt, end="", flush=True)

  for _ in range(max_new_tokens):
    context = model_input[:, -model.context_len:]

    next_token = model.predict(context, top_k=4, temperature=0.5)
    model_input = np.concatenate([model_input, next_token[:, None]], axis=1)
    att_scores = model.layers[2].attn.forward(model.layers[1].forward(model.layers[0].forward(context)))
    next_char = tokenizer.decode(next_token[0:1])
    print(next_char, end="",flush=True)

  print()


model = Model.load("models/2025-07-21_22-41-02")

prompt = "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei"
generate_text_streaming(model, tokenizer, prompt, max_new_tokens=422)