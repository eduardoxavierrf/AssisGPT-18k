import json
from typing import List, Optional
from collections import Counter
import re

class CharTokenizer:
  def __init__(self, special_tokens: Optional[List[str]] = None, max_vocab_size: int = 65):
    self.char_to_idx = {}
    self.idx_to_char = {}
    self.vocab_size = 0
    self._is_fitted = False
    self.max_vocab_size = max_vocab_size
    
    if special_tokens is None:
      special_tokens = ['<UNK>']
    
    self.add_special_tokens(special_tokens)
  
  def fit(self, texts: List[str]) -> None:
    char_counter = Counter()
    for text in texts:
      char_counter.update(text)
    
    available_slots = self.max_vocab_size - len(self.char_to_idx)
    most_frequent_chars = [char for char, count in char_counter.most_common(available_slots) 
                          if char not in self.char_to_idx]
    
    for char in most_frequent_chars:
      self.char_to_idx[char] = self.vocab_size
      self.idx_to_char[self.vocab_size] = char
      self.vocab_size += 1
    
    self._is_fitted = True
  
  def encode(self, text: str) -> List[int]:
    if not self._is_fitted:
      raise ValueError("Tokenizer must be fitted before encoding")
    
    unk_token = self.char_to_idx.get('<UNK>')
    if unk_token is None:
      raise ValueError("No <UNK> token found in vocabulary")
    
    return [self.char_to_idx.get(char, unk_token) for char in text]
  
  def decode(self, tokens: List[int]) -> str:
    return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in tokens])
  
  def add_special_tokens(self, tokens: List[str]) -> None:
    for token in tokens:
      if token not in self.char_to_idx:
        self.char_to_idx[token] = self.vocab_size
        self.idx_to_char[self.vocab_size] = token
        self.vocab_size += 1
  
  def is_fitted(self) -> bool:
    return self._is_fitted
  
  def get_vocab(self) -> dict:
    return self.char_to_idx.copy()
  
  def get_vocab_size(self) -> int:
    return self.vocab_size
  
  def save(self, filepath: str) -> None:
    data = {
      'char_to_idx': self.char_to_idx,
      'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
      'vocab_size': self.vocab_size,
      'is_fitted': self._is_fitted
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
  
  @staticmethod
  def load(filepath: str, max_vocab_size: int = 65) -> None:
    with open(filepath, 'r', encoding='utf-8') as f:
      data = json.load(f)
    
    tokenizer = CharTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.char_to_idx = data['char_to_idx']
    tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
    tokenizer.vocab_size = data['vocab_size']
    tokenizer._is_fitted = data.get('is_fitted', True)

    return tokenizer
  
  def __len__(self) -> int:
    return self.vocab_size
  
  def __repr__(self) -> str:
    return f"CharTokenizer(vocab_size={self.vocab_size}, fitted={self._is_fitted})"